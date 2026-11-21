"""Step 06: Bayesian Poisson model with BYM2 municipal effects, by sex.

For sex s (F, M), municipality i and year t:
    y_sit ~ Poisson(E_sit * theta_sit)
    log theta_sit = alpha_s + x_i' beta_s + b_si + gamma_st
- E_sit: expected suicides from the national age-specific rates of sex s in year t (step 03), so
  theta_sit is the relative risk against the national level of the same sex and year.
- x_i: standardised municipal covariates (step 04); model A = IPM, model B = the five dimensions.
- b_si: BYM2 (Riebler et al., 2016) on the national contiguity graph (step 05), separate for each sex:
  sigma_bs * (sqrt(rho_s / c) * phi_si + sqrt(1 - rho_s) * u_si), phi ~ ICAR (soft sum to zero),
  u ~ N(0, 1), c = scaling factor of the graph.
- gamma_st: first-order random walk by sex (centred).
All parameters are sex-specific, so the joint fit gives the same posterior for each sex as two separate
fits, and the women/men ratio of relative risks exp(beta_F - beta_M) is computed draw by draw.
Priors: alpha ~ N(0, 1); beta ~ N(0, 1); sigma ~ Exponential(4.6); rho ~ Beta(0.5, 0.5).
NUTS, 4 chains x (4000 warm-up + 12000 draws), seed 20261118 (shorter chains left R-hat or ESS of rho above
the thresholds).

Options (pre-specified sensitivities): --source {eevv_residence, eevv_extended, inmlcf_occurrence},
--negbin, --nonspatial, --years 2018-2024, --covs (model A: ipm; NBI sensitivity: nbi; model B: list).
Additional sensitivity options: --rho-prior uniform (Beta(1, 1) instead of Beta(0.5, 0.5));
--adjacency knn6 (graph of the six nearest municipalities, step 05b); --nb-prior halfnormal (negative binomial with
inv_k ~ HalfNormal(0.25), a more informative prior on overdispersion). Every run also reports posterior predictive
coverage and zeros by stratum of expected counts, and municipal totals over the period.

Outputs: results/model/<tag>/ summary.json, posterior.npz, rr_municipal.csv

Usage (from the repository root):
  python src/06_model.py --covs ipm --tag A_main
  python src/06_model.py --covs B --tag B_main
"""
import argparse
import json
import time
from pathlib import Path

import numpyro

numpyro.set_host_device_count(4)

import arviz as az  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from numpyro.infer import MCMC, NUTS, init_to_value  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SEED = 20261118
SEXES = ["F", "M"]


def scaling_factor(n, edges):
    """Geometric mean of the marginal variances of the sum-to-zero constrained ICAR (connected graph)."""
    Q = np.zeros((n, n))
    for i, j in edges:
        Q[i, j] = Q[j, i] = -1
    np.fill_diagonal(Q, -Q.sum(axis=1))
    Qp = Q + np.eye(n) * Q.diagonal().max() * np.sqrt(np.finfo(float).eps)
    S = np.linalg.inv(Qp)
    A = np.ones((1, n))
    S = S - S @ A.T @ np.linalg.inv(A @ S @ A.T) @ A @ S
    return float(np.exp(np.mean(np.log(np.diag(S)))))


def load(source, years, covs):
    cov = pd.read_csv(PROC / "covariates.csv", dtype={"muni": str})
    order = cov["muni"].tolist()
    p = pd.read_csv(PROC / f"panel_{source}.csv", dtype={"muni": str})
    p = p[p.year.isin(years)]
    Y = np.stack([p[p.sex == s].pivot(index="muni", columns="year", values="y").loc[order].to_numpy() for s in SEXES])
    E = np.stack([p[p.sex == s].pivot(index="muni", columns="year", values="E").loc[order].to_numpy() for s in SEXES])
    X = cov[["z_" + c for c in covs]].to_numpy(float)
    return Y.astype(np.int32), E, X, order, cov


def model(E, X, ei, ej, c, spatial, negbin, Y=None, rho_a=0.5, nb_halfnormal=False):
    S, n, T = E.shape
    K = X.shape[1]
    alpha = numpyro.sample("alpha", dist.Normal(0, 1).expand([S]))
    beta = numpyro.sample("beta", dist.Normal(0, 1).expand([S, K]))
    sigma_b = numpyro.sample("sigma_b", dist.Exponential(4.6).expand([S]))
    u = numpyro.sample("u", dist.Normal(0, 1).expand([S, n]))
    if spatial:
        rho = numpyro.sample("rho", dist.Beta(rho_a, rho_a).expand([S]))
        phi = numpyro.sample("phi", dist.ImproperUniform(dist.constraints.real, (), (S, n)))
        numpyro.factor("icar", -0.5 * jnp.sum((phi[:, ei] - phi[:, ej]) ** 2))
        numpyro.factor("phi_sum0", jnp.sum(dist.Normal(0, 0.001 * n).log_prob(jnp.sum(phi, axis=1))))
        b = sigma_b[:, None] * (jnp.sqrt(rho[:, None] / c) * phi + jnp.sqrt(1 - rho[:, None]) * u)
    else:
        b = sigma_b[:, None] * u
    numpyro.deterministic("b", b)
    sigma_g = numpyro.sample("sigma_g", dist.Exponential(4.6).expand([S]))
    g_raw = numpyro.sample("g_raw", dist.Normal(0, 1).expand([S, T]))
    g = jnp.cumsum(g_raw, axis=1) * sigma_g[:, None]
    g = g - g.mean(axis=1, keepdims=True)
    numpyro.deterministic("gamma", g)
    eta = alpha[:, None, None] + (beta @ X.T)[:, :, None] + b[:, :, None] + g[:, None, :]
    mu = E * jnp.exp(eta)
    if negbin:
        inv_k = numpyro.sample("inv_k", (dist.HalfNormal(0.25) if nb_halfnormal else dist.Exponential(4.6)).expand([S]))
        numpyro.sample("y", dist.NegativeBinomial2(mu, 1.0 / (inv_k[:, None, None] + 1e-6)), obs=Y)
    else:
        numpyro.sample("y", dist.Poisson(mu), obs=Y)


def q(x, axis=0):
    return np.quantile(x, [0.5, 0.025, 0.975], axis=axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="eevv_residence",
                    choices=["eevv_residence", "eevv_extended", "inmlcf_occurrence"])
    ap.add_argument("--covs", default="ipm", help="ipm | nbi | B | comma-separated list")
    ap.add_argument("--years", default="2015-2024")
    ap.add_argument("--negbin", action="store_true")
    ap.add_argument("--nonspatial", action="store_true")
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--draws", type=int, default=12000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--progress", action="store_true", help="show the sampler progress bar (written to the log)")
    ap.add_argument("--max-tree-depth", type=int, default=10, help="NUTS maximum tree depth (default 10)")
    ap.add_argument("--rho-prior", default="beta05", choices=["beta05", "uniform"])
    ap.add_argument("--adjacency", default="main", choices=["main", "knn6"])
    ap.add_argument("--nb-prior", default="exponential", choices=["exponential", "halfnormal"])
    a = ap.parse_args()
    t0 = time.time()
    y0, y1 = map(int, a.years.split("-"))
    years = list(range(y0, y1 + 1))
    if a.covs == "B":
        covs = json.loads((ROOT / "results" / "covariates" / "collinearity.json")
                          .read_text(encoding="utf-8"))["model_b_final"]
    else:
        covs = a.covs.split(",")
    out = ROOT / "results" / "model" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    Y, E, X, order, cov = load(a.source, years, covs)
    S, n, T = Y.shape
    adj = pd.read_csv(PROC / ("adjacency.csv" if a.adjacency == "main" else "adjacency_knn6.csv"))
    ei, ej = adj.node1.to_numpy() - 1, adj.node2.to_numpy() - 1
    c = scaling_factor(n, list(zip(ei, ej)))
    spatial = not a.nonspatial

    # neutral starting values (all effects 0, small SDs): the default uniform(-2, 2) start puts expected counts
    # of large municipalities orders of magnitude off and the chains did not move 
    K = X.shape[1]
    init = {"alpha": jnp.zeros(S), "beta": jnp.zeros((S, K)), "sigma_b": jnp.full(S, 0.3), "u": jnp.zeros((S, n)),
            "sigma_g": jnp.full(S, 0.1), "g_raw": jnp.zeros((S, T))}
    if spatial:
        init |= {"rho": jnp.full(S, 0.5), "phi": jnp.zeros((S, n))}
    if a.negbin:
        init["inv_k"] = jnp.full(S, 0.1)
    kernel = NUTS(model, target_accept_prob=0.9, init_strategy=init_to_value(values=init),
                  max_tree_depth=a.max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=a.warmup, num_samples=a.draws, num_chains=4,
                chain_method="parallel", progress_bar=a.progress)
    mcmc.run(jax.random.PRNGKey(SEED), E=jnp.array(E), X=jnp.array(X), ei=jnp.array(ei), ej=jnp.array(ej), c=c,
             spatial=spatial, negbin=a.negbin, Y=jnp.array(Y), rho_a=0.5 if a.rho_prior == "beta05" else 1.0,
             nb_halfnormal=a.nb_prior == "halfnormal")
    idata = az.from_numpyro(mcmc)
    smp = mcmc.get_samples()
    beta = np.asarray(smp["beta"])               # draws x S x K
    b = np.asarray(smp["b"])                     # draws x S x n
    names = ["alpha", "beta", "sigma_b", "sigma_g", "u", "g_raw"] + (["rho", "phi"] if spatial else []) \
        + (["inv_k"] if a.negbin else [])
    rh = az.rhat(idata, var_names=names)
    es = az.ess(idata, var_names=names)
    diag = {"max_rhat": float(max(float(v.max()) for v in rh.data_vars.values())),
            "min_ess_bulk": float(min(float(v.min()) for v in es.data_vars.values())),
            "divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
            "max_rhat_by_param": {k: round(float(v.max()), 4) for k, v in rh.data_vars.items()},
            "min_ess_by_param": {k: round(float(v.min()), 1) for k, v in es.data_vars.items()}}

    # posterior predictive checks (zeros, dispersion of y/E, coverage), by sex
    rng = np.random.default_rng(SEED)
    sel = rng.choice(beta.shape[0], size=min(1000, beta.shape[0]), replace=False)
    eta = (np.asarray(smp["alpha"])[sel][:, :, None, None] + np.einsum("dsk,nk->dsn", beta[sel], X)[:, :, :, None]
           + b[sel][:, :, :, None] + np.asarray(smp["gamma"])[sel][:, :, None, :])
    mu = E[None] * np.exp(eta)
    if a.negbin:
        k = 1.0 / (np.asarray(smp["inv_k"])[sel][:, :, None, None] + 1e-6)
        yrep = rng.poisson(rng.gamma(k, mu / k))
    else:
        yrep = rng.poisson(mu)
    ppc = {}
    for si, s in enumerate(SEXES):
        yo, yr, Es = Y[si], yrep[:, si], E[si]
        ok = Es > 0
        d_obs = np.var(yo[ok] / Es[ok]) / np.mean(yo[ok] / Es[ok])
        d_rep = np.array([np.var(r[ok] / Es[ok]) / np.mean(r[ok] / Es[ok]) for r in yr])
        z_rep = (yr == 0).sum(axis=(1, 2))
        ppc[s] = {"zeros_obs": int((yo == 0).sum()), "zeros_rep_mean": float(z_rep.mean()),
                  "ppp_zeros": float((z_rep >= (yo == 0).sum()).mean()), "ppp_dispersion": float((d_rep >= d_obs).mean()),
                  "coverage_90": float(((yo >= np.quantile(yr, 0.05, 0)) & (yo <= np.quantile(yr, 0.95, 0))).mean())}
        # by stratum of expected counts of the municipality-year cell
        lo, hi = np.quantile(yr, 0.05, 0), np.quantile(yr, 0.95, 0)
        strata = {}
        for name, m in (("E<0.5", Es < 0.5), ("0.5<=E<2", (Es >= 0.5) & (Es < 2)), ("2<=E<10", (Es >= 2) & (Es < 10)),
                        ("E>=10", Es >= 10)):
            if m.sum() == 0:
                continue
            strata[name] = {"cells": int(m.sum()), "coverage_90": float(((yo >= lo) & (yo <= hi))[m].mean()),
                            "zeros_obs": int((yo[m] == 0).sum()), "zeros_rep_mean": float((yr[:, m] == 0).sum(1).mean()),
                            "width_90_mean": float((hi - lo)[m].mean())}
        ppc[s]["by_expected_stratum"] = strata
        # municipal totals over the whole period
        tot_o, tot_r = yo.sum(1), yr.sum(2)
        tlo, thi = np.quantile(tot_r, 0.05, 0), np.quantile(tot_r, 0.95, 0)
        ppc[s]["municipal_total_coverage_90"] = float(((tot_o >= tlo) & (tot_o <= thi)).mean())
        ppc[s]["max_count_obs"] = int(yo.max())
        ppc[s]["ppp_max_count"] = float((yr.max(axis=(1, 2)) >= yo.max()).mean())

    res = {}
    for k, cname in enumerate(covs):
        rr = np.exp(beta[:, :, k])
        ratio = np.exp(beta[:, 0, k] - beta[:, 1, k])
        res[cname] = {s: dict(zip(["rr_median", "rr_lo", "rr_hi"], q(rr[:, si]).round(4).tolist()))
                      | {"p_rr_gt1": round(float((rr[:, si] > 1).mean()), 4)} for si, s in enumerate(SEXES)}
        res[cname]["ratio_F_over_M"] = dict(zip(["median", "lo", "hi"], q(ratio).round(4).tolist())) \
            | {"p_gt1": round(float((ratio > 1).mean()), 4)}
    hyper = {}
    for name in ["alpha", "sigma_b", "sigma_g"] + (["rho"] if spatial else []) + (["inv_k"] if a.negbin else []):
        v = np.asarray(smp[name])
        hyper[name] = {s: q(v[:, si]).round(4).tolist() for si, s in enumerate(SEXES)}

    rows = []
    for si, s in enumerate(SEXES):
        rr = np.exp(b[:, si])
        lo_med_hi = q(rr)
        rows.append(pd.DataFrame({"muni": order, "sex": s, "rr_median": lo_med_hi[0], "rr_lo": lo_med_hi[1],
                                  "rr_hi": lo_med_hi[2], "p_gt1": (rr > 1).mean(0),
                                  "y": Y[si].sum(1), "E": E[si].sum(1)}))
    pd.concat(rows).round(5).to_csv(out / "rr_municipal.csv", index=False)
    # all draws for beta and hyperparameters (in chain order); municipal effects thinned to every 10th draw
    np.savez_compressed(out / "posterior.npz", beta=beta, b_thin10=b[::10], covs=np.array(covs), munis=np.array(order),
                        sexes=np.array(SEXES), **{k: np.asarray(smp[k]) for k in hyper})
    summary = {"tag": a.tag, "source": a.source, "covariates": covs, "years": years, "spatial": spatial,
               "negbin": a.negbin, "n_areas": n, "n_years": T,
               "y_total": {s: int(Y[si].sum()) for si, s in enumerate(SEXES)},
               "E_total": {s: round(float(E[si].sum()), 2) for si, s in enumerate(SEXES)},
               "scaling_factor": c, "rho_prior": a.rho_prior, "adjacency": a.adjacency, "nb_prior": a.nb_prior,
               "divergences_by_chain": np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum(axis=1).tolist(),
               "chains": 4, "warmup": a.warmup, "draws": a.draws, "max_tree_depth": a.max_tree_depth, "seed": SEED,
               "diagnostics": diag, "ppc": ppc, "rr_per_sd": res, "hyper_median_lo_hi": hyper,
               "runtime_min": round((time.time() - t0) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("tag", "diagnostics", "rr_per_sd", "runtime_min")}, indent=1))


if __name__ == "__main__":
    main()
