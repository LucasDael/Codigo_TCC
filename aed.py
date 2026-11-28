from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from config import OUTPUT_PATH

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110

REPORTS_DIR = Path("eda_reports")
REPORTS_DIR.mkdir(exist_ok=True)

TARGET_COL = "target"
ID_COLS = ["fixture_id", "date", "league_id", "league_season",
           "teams_home_id", "teams_away_id"]


# --------------------------------------------------------------------------
# 1. Sanidade estrutural
# --------------------------------------------------------------------------
def report_structure(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("1. ESTRUTURA")
    print("=" * 70)
    print(f"Shape: {df.shape[0]:,} linhas x {df.shape[1]:,} colunas")
    print(f"Memoria: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    dup_fixture = df["fixture_id"].duplicated().sum()
    print(f"fixture_id duplicados: {dup_fixture}")

    dtypes = df.dtypes.value_counts()
    print(f"\nTipos:\n{dtypes.to_string()}")

    n_obj = (df.dtypes == object).sum()
    if n_obj:
        print(f"\nAVISO: {n_obj} colunas object — checar se sao datas/ids.")

    print(f"\nIntervalo temporal: {df['date'].min()} -> {df['date'].max()}")
    print(f"Ligas: {df['league_id'].nunique()}  |  "
          f"Temporadas: {df['league_season'].nunique()}")
    print(f"Equipes (mandantes): {df['teams_home_id'].nunique()}  |  "
          f"(visitantes): {df['teams_away_id'].nunique()}")


# --------------------------------------------------------------------------
# 2. Alvo e gols
# --------------------------------------------------------------------------
def report_target(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("2. ALVO E GOLS")
    print("=" * 70)
    vc = df[TARGET_COL].value_counts().sort_index()
    pct = df[TARGET_COL].value_counts(normalize=True).sort_index()
    labels = {0: "Empate", 1: "Casa", 2: "Fora"}
    for k in vc.index:
        print(f"  {labels[k]:<8} = {k}: {vc[k]:>7,} ({pct[k]*100:5.2f}%)")

    # Checagem de balanceamento (chi2 contra uniforme)
    chi2, p = stats.chisquare(vc.values)
    p_display = f"p={p:.2e}" if p > 0 else "p<1e-308 (limite numerico)"
    print(f"\nChi2 balanceamento: chi2={chi2:.1f}, {p_display} "
          f"({'balanceado' if p > 0.01 else 'DESbalanceado'})")


def plot_target_and_lambdas(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    counts = df[TARGET_COL].map({0: "Empate", 1: "Casa", 2: "Fora"}).value_counts()
    axes[0, 0].bar(counts.index, counts.values, color=["#888", "#3b82f6", "#ef4444"])
    axes[0, 0].set_title("Distribuicao do alvo (1X2)")

    if "lambda_home" in df:
        axes[0, 1].hist(df["lambda_home"].dropna(), bins=50, alpha=0.7,
                        label="lambda_home", color="#3b82f6")
        axes[0, 1].hist(df["lambda_away"].dropna(), bins=50, alpha=0.7,
                        label="lambda_away", color="#ef4444")
        axes[0, 1].set_title("Lambdas de Poisson")
        axes[0, 1].legend()

    prob_cols = [c for c in df.columns if c.startswith("poisson_prob_")]
    if prob_cols:
        df[prob_cols].hist(ax=axes[1, 0], bins=40, alpha=0.6)
        axes[1, 0].set_title("Probabilidades de Poisson")

    # Alvo vs. prob prevista — calibracao grosseira
    if "poisson_prob_home_win" in df:
        df_plot = df.copy()
        df_plot["bucket"] = pd.cut(df_plot["poisson_prob_home_win"], bins=10)
        cal = df_plot.groupby("bucket", observed=True).agg(
            pred=("poisson_prob_home_win", "mean"),
            real=(TARGET_COL, lambda s: (s == 1).mean()),
            n=("fixture_id", "count"),
        )
        axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[1, 1].scatter(cal["pred"], cal["real"], s=cal["n"] / 10, alpha=0.8)
        axes[1, 1].set_xlabel("Prob casa prevista (Poisson)")
        axes[1, 1].set_ylabel("Frequencia real casa")
        axes[1, 1].set_title("Calibracao 1x1 (Poisson)")

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "01_target_e_poisson.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Qualidade das features
# --------------------------------------------------------------------------
def report_feature_quality(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("3. QUALIDADE DAS FEATURES")
    print("=" * 70)
    feat_cols = [c for c in df.columns if c not in ID_COLS + [TARGET_COL]]

    summary = pd.DataFrame({
        "dtype": df[feat_cols].dtypes.astype(str),
        "n_null": df[feat_cols].isna().sum(),
        "pct_null": df[feat_cols].isna().mean() * 100,
        "n_unique": df[feat_cols].nunique(dropna=True),
        "std": df[feat_cols].std(numeric_only=True),
    }).sort_values("pct_null", ascending=False)

    # Constantes
    constantes = summary[summary["n_unique"] <= 1]
    print(f"Colunas CONSTANTES (n_unique<=1): {len(constantes)}")
    if len(constantes):
        print(constantes[["n_unique", "pct_null"]].head(20).to_string())

    # Quase-constantes (variancia ~0)
    qc = summary[(summary["std"].fillna(0) < 1e-8) &
                 (summary["n_unique"] > 1)]
    print(f"\nColunas quase-constantes (std<1e-8): {len(qc)}")
    if len(qc):
        print(qc[["n_unique", "std"]].head(20).to_string())

    # Altos nulos
    high_null = summary[summary["pct_null"] > 60]
    print(f"\nColunas com >60% nulos: {len(high_null)}")
    if len(high_null):
        print(high_null[["pct_null", "n_unique"]].head(25).to_string())

    summary.to_csv(REPORTS_DIR / "feature_quality.csv")
    print(f"\nRelatorio completo: {REPORTS_DIR/'feature_quality.csv'}")
    return summary


def plot_null_heatmap(df: pd.DataFrame, max_cols: int = 80) -> None:
    """Heatmap de nulos nas top-N colunas com mais nulos."""
    null_pct = df.isna().mean().sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0].head(max_cols)
    if null_pct.empty:
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(null_pct) * 0.15)))
    sns.barplot(x=null_pct.values * 100, y=null_pct.index,
                palette="Reds_r", ax=ax)
    ax.set_xlabel("% de nulos")
    ax.set_title(f"Top {len(null_pct)} colunas por % de nulos")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "02_nulos_top.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Correlacao com o alvo e multicolinearidade
# --------------------------------------------------------------------------
def report_correlations(df: pd.DataFrame, top_n: int = 30) -> None:
    print("\n" + "=" * 70)
    print("4. CORRELACAO E MULTICOLINEARIDADE")
    print("=" * 70)
    feat_cols = [c for c in df.columns
                 if c not in ID_COLS and df[c].dtype != object]

    # Correlacao univariada com target (apenas binario casa vs nao-casa
    # e casa vs fora para leitura rapida).
    df_bin = df.copy()
    corr_target = {}
    for col in feat_cols:
        if col == TARGET_COL:
            continue
        try:
            r, _ = stats.pointbiserialr(
                (df_bin[TARGET_COL] == 1).astype(int), df_bin[col].fillna(df_bin[col].median())
            )
            corr_target[col] = r
        except Exception:
            corr_target[col] = np.nan

    corr_series = pd.Series(corr_target).dropna().sort_values(key=np.abs, ascending=False)
    print(f"\nTop {top_n} features por |correlacao| com P(casa vence):")
    print(corr_series.head(top_n).to_string())

    # Multicolinearidade: pega top-30 features e calcula matriz
    top_feats = corr_series.head(30).index.tolist()
    if len(top_feats) > 1:
        cm = df[top_feats].corr()
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(cm, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    square=True, ax=ax, cbar_kws={"shrink": 0.7})
        ax.set_title("Correlacao entre top-30 features (com target)")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "03_corr_top30.png", bbox_inches="tight")
        plt.close(fig)

        # Pares altamente correlacionados
        upper = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
        high_corr = upper.stack().sort_values(key=np.abs, ascending=False)
        high_corr = high_corr[high_corr.abs() > 0.95]
        print(f"\nPares com |corr| > 0.95: {len(high_corr)}")
        if len(high_corr):
            print(high_corr.head(20).to_string())

    corr_series.to_csv(REPORTS_DIR / "corr_com_target.csv", header=["r"])


# --------------------------------------------------------------------------
# 5. Leakage temporal
# --------------------------------------------------------------------------
def report_leakage(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("5. LEAKAGE TEMPORAL")
    print("=" * 70)

    # Features cujo nome sugere dependencia do resultado atual
    suspeitas = [c for c in df.columns if any(
        k in c.lower() for k in
        ["goals_home", "goals_away", "score_", "winner", "target"]
    ) and c != TARGET_COL]
    print(f"Colunas com nome suspeito de leakage: {len(suspeitas)}")
    if suspeitas:
        print(suspeitas[:30])

    # Checagem empirica: correlação quase perfeita com target
    feat_cols = [c for c in df.columns
                 if c not in ID_COLS and df[c].dtype != object]
    df_bin = df.copy()
    corrs = {}
    for col in feat_cols:
        if col == TARGET_COL:
            continue
        try:
            r, _ = stats.pointbiserialr(
                (df_bin[TARGET_COL] == 1).astype(int),
                df_bin[col].fillna(df_bin[col].median()),
            )
            corrs[col] = abs(r)
        except Exception:
            pass
    corrs = pd.Series(corrs).sort_values(ascending=False)
    suspeitas_emp = corrs[corrs > 0.9]
    print(f"\nFeatures com |corr| > 0.9 com o alvo (leakage provavel): "
          f"{len(suspeitas_emp)}")
    if len(suspeitas_emp):
        print(suspeitas_emp.head(20).to_string())


# --------------------------------------------------------------------------
# 6. Diagnostico das forcas de Poisson
# --------------------------------------------------------------------------
def report_poisson_strengths(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("6. FORCAS E LAMBDAS DE POISSON")
    print("=" * 70)
    for side in ["home", "away"]:
        col_att = f"{side}_attack_strength"
        col_def = f"{side}_defense_concession_strength"
        if col_att in df:
            s = df[col_att].dropna()
            print(f"{col_att:<40} min={s.min():.2f} p50={s.median():.2f} "
                  f"p95={s.quantile(.95):.2f} max={s.max():.2f} nulos={df[col_att].isna().sum()}")
        if col_def in df:
            s = df[col_def].dropna()
            print(f"{col_def:<40} min={s.min():.2f} p50={s.median():.2f} "
                  f"p95={s.quantile(.95):.2f} max={s.max():.2f} nulos={df[col_def].isna().sum()}")

    # Sanidade: lambdas negativos ou absurdos
    for col in ["lambda_home", "lambda_away"]:
        if col in df:
            s = df[col].dropna()
            neg = (s <= 0).sum()
            absurd = (s > 6).sum()
            print(f"{col:<12} negativos={neg}  >6={absurd}  "
                  f"p99={s.quantile(.99):.2f}  max={s.max():.2f}")

    # Amostra de consistencia: lambda_home + lambda_away vs media de gols
    if {"lambda_home", "lambda_away"}.issubset(df.columns):
        tot_lambda = (df["lambda_home"] + df["lambda_away"]).dropna()
        print(f"\nSoma de lambdas (gols esperados/jogo): "
              f"media={tot_lambda.mean():.2f}  "
              f"p50={tot_lambda.median():.2f}  "
              f"p99={tot_lambda.quantile(.99):.2f}")


# --------------------------------------------------------------------------
# 7. Estabilidade temporal
# --------------------------------------------------------------------------
def report_temporal(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("7. ESTABILIDADE TEMPORAL")
    print("=" * 70)
    df_t = df.sort_values("date").copy()
    df_t["ano"] = pd.to_datetime(df_t["date"]).dt.year

    por_ano = df_t.groupby("ano").agg(
        n=(TARGET_COL, "count"),
        pct_casa=(TARGET_COL, lambda s: (s == 1).mean()),
        pct_empate=(TARGET_COL, lambda s: (s == 0).mean()),
        pct_fora=(TARGET_COL, lambda s: (s == 2).mean()),
    )
    print(por_ano.round(3).to_string())

    # Cobertura temporal das features (a primeira temporada deve ter nulos)
    if "home_all_goals_for_mean_all" in df.columns:
        cobertura = (df_t.groupby("ano")["home_all_goals_for_mean_all"]
                     .apply(lambda s: s.notna().mean()))
        print(f"\nCobertura da media historica do mandante por ano:")
        print((cobertura * 100).round(1).astype(str).add(" %").to_string())


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    print(f"Carregando {OUTPUT_PATH}...")
    df = pd.read_csv(OUTPUT_PATH, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    report_structure(df)
    report_target(df)
    plot_target_and_lambdas(df)
    report_feature_quality(df)
    plot_null_heatmap(df)
    report_correlations(df)
    report_leakage(df)
    report_poisson_strengths(df)
    report_temporal(df)

    # Matriz de correlacao geral para colunas com baixo nulo
    feat_ok = [c for c in df.columns if df[c].isna().mean() < 0.3
               and df[c].dtype != object and c != TARGET_COL]
    if len(feat_ok) > 1:
        cm = df[feat_ok[:50]].corr()
        fig, ax = plt.subplots(figsize=(13, 11))
        sns.heatmap(cm, cmap="RdBu_r", center=0, ax=ax,
                    vmin=-1, vmax=1, cbar_kws={"shrink": 0.7})
        ax.set_title(f"Correlacao entre {min(50, len(feat_ok))} features "
                     f"com <30% nulos")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "04_corr_features.png", bbox_inches="tight")
        plt.close(fig)

    print(f"\nGraficos salvos em {REPORTS_DIR.resolve()}/")


if __name__ == "__main__":
    main()