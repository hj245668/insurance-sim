"""보험수리 시뮬레이터 (Streamlit). 교육용이며 실제 보험료가 아닙니다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from insurance_sim.life import (
    DEMO_TABLE, annuity_due_apv, endowment_apv, net_level_premium,
    prospective_reserve, term_insurance_apv,
)
from insurance_sim.nonlife import (
    compound_poisson_moments, gross_premium, lundberg_exponential, normal_premium,
)
from insurance_sim.simulation import simulate_aggregate_loss, simulate_ruin_probability

T = DEMO_TABLE
st.set_page_config(page_title="보험수리 시뮬레이터", page_icon="📘", layout="wide")
st.title("보험수리 시뮬레이터")
st.caption("가상 생명표와 단순화된 가정을 쓴 교육용 도구입니다. 실제 보험료가 아닙니다.")

tab1, tab2, tab3 = st.tabs(["생명보험 보험료", "손해보험 모듈", "파산확률"])


@st.cache_data
def run_loss_sim(n_pol, lam, mean, sd, seed):
    return simulate_aggregate_loss(n_pol, lam, mean, sd, n_trials=10000, seed=seed)


@st.cache_data
def run_ruin_sim(u, theta, mu, seed):
    return simulate_ruin_probability(u, theta, mu, n_paths=5000, n_claims=300, seed=seed)


with tab1:
    c = st.columns(5)
    age = c[0].selectbox("가입 나이", [60, 61, 62])
    n = c[1].slider("보험 기간(년)", 1, T.max_age - age, T.max_age - age)
    kind_name = c[2].radio("종류", ["정기보험", "양로보험"], horizontal=True)
    i = c[3].slider("이자율(%)", 0.0, 8.0, 3.0, 0.1) / 100
    benefit = c[4].number_input("보험금(원)", 1_000_000, 1_000_000_000, 100_000_000, 10_000_000)
    expense = st.slider("사업비율(%)", 0, 50, 20) / 100
    kind = "term" if kind_name == "정기보험" else "endowment"

    apv = (term_insurance_apv if kind == "term" else endowment_apv)(T, age, n, i, benefit)
    adue = annuity_due_apv(T, age, n, i)
    p = net_level_premium(T, age, n, i, benefit, kind)
    m = st.columns(4)
    m[0].metric("연납 순보험료", f"{p:,.0f}원")
    m[1].metric("연납 총보험료", f"{gross_premium(p, expense):,.0f}원")
    m[2].metric("일시납 순보험료", f"{apv:,.0f}원")
    m[3].metric("납입 현가 ä", f"{adue:.4f}")
    st.caption("등가원리: 연납 순보험료 × ä = 보험금의 기대현가")

    left, right = st.columns(2)
    with left:
        st.subheader("연도별 책임준비금")
        res = pd.DataFrame(
            {"경과 연수": list(range(n)),
             "준비금(원)": [prospective_reserve(T, age, n, i, t, benefit, kind) for t in range(n)]}
        ).set_index("경과 연수")
        st.bar_chart(res)
    with right:
        st.subheader("사용한 생명표")
        st.dataframe(
            pd.DataFrame({"나이": list(range(age, age + n)),
                          "l_x": [T.l(a) for a in range(age, age + n)],
                          "q_x": [round(T.q(a), 5) for a in range(age, age + n)]}),
            hide_index=True,
        )

with tab2:
    c = st.columns(3)
    lam = c[0].slider("연간 사고 빈도 λ", 0.01, 1.0, 0.1, 0.01)
    mean = c[1].number_input("건당 평균 손해(만원)", 10, 5000, 200, 10)
    sd = c[2].number_input("건당 손해 표준편차(만원)", 10, 5000, 100, 10)
    c = st.columns(3)
    n_pol = c[0].number_input("계약 수", 100, 20000, 1000, 100)
    exp2 = c[1].slider("사업비율(%) ", 0, 50, 20) / 100
    conf = c[2].slider("신뢰수준(%)", 90.0, 99.9, 99.0, 0.1) / 100

    pure = lam * mean
    normal = normal_premium(lam, mean, sd, n_pol, conf)
    losses = run_loss_sim(int(n_pol), lam, mean, sd, 1)
    mc = float(np.quantile(losses, conf)) / n_pol
    mean_t, var_t = compound_poisson_moments(lam, mean, sd, n_pol)
    m = st.columns(4)
    m[0].metric("계약당 순보험료(만원)", f"{pure:,.2f}")
    m[1].metric("계약당 총보험료(만원)", f"{gross_premium(pure, exp2):,.2f}")
    m[2].metric(f"{conf:.1%} 수준 보험료: 정규근사", f"{normal:,.2f}")
    m[3].metric(f"{conf:.1%} 수준 보험료: 시뮬레이션", f"{mc:,.2f}")
    st.caption(f"총손해 이론 평균 {mean_t:,.0f}만원, 표준편차 {var_t ** 0.5:,.0f}만원. 시뮬레이션이 정규근사보다 크면 분포가 오른쪽으로 치우쳤다는 뜻입니다.")
    counts, edges = np.histogram(losses, bins=40)
    st.bar_chart(pd.DataFrame({"횟수": counts}, index=[f"{e:,.0f}" for e in edges[:-1]]))

with tab3:
    c = st.columns(3)
    theta = c[0].slider("안전할증률 θ", 0.05, 1.0, 0.2, 0.05)
    mu = c[1].number_input("평균 손해 μ(만원)", 10, 5000, 200, 10)
    u0 = c[2].number_input("초기 자본 u(만원)", 0, 100000, 2000, 100)
    psi = lundberg_exponential(u0, theta, mu)
    st.metric("파산확률 ψ(u) (Lundberg 공식)", f"{psi:.2%}")
    us = np.linspace(0, max(5 * mu * 10, u0 * 2, 1), 60)
    curve = pd.DataFrame(
        {f"θ={t:.2f}": [lundberg_exponential(x, t, mu) for x in us]
         for t in sorted({round(theta / 2, 2), theta, round(theta * 2, 2)})},
        index=us.round(0),
    )
    st.line_chart(curve)
    if st.button("시뮬레이션으로 확인"):
        est = run_ruin_sim(float(u0), theta, float(mu), 7)
        st.write(f"시뮬레이션 추정 {est:.2%} (경로 5,000개, 공식 {psi:.2%})")
