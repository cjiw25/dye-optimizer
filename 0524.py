import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.interpolate import interp1d

# 1. 페이지 설정
st.set_page_config(page_title="AI Dye Optimizer", page_icon="🎨", layout="wide")
st.title("🎨 염료 농도 최적화 프로그램")

# 2. 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv("0524data.csv")
    return df

df = load_data()

# 3. K/S 변환 함수
def calc_ks(r_val):
    r = np.array(r_val, dtype=float) / 100.0
    r = np.clip(r, 0.0001, 1.0)
    return ((1 - r)**2) / (2 * r)

# 4. K/S -> RGB Hex 변환 함수 (색상 패치용)
def ks_to_hex(ks_array):
    v = np.array(ks_array).flatten()
    R = (v + 1) - np.sqrt(v**2 + 2*v)
    R = np.clip(R, 0, 1)

    cmf_x = np.array([0.0143, 0.0435, 0.1344, 0.2839, 0.3483, 0.3362, 0.2908, 0.1954, 0.0956, 0.0320, 0.0049, 0.0093, 0.0633, 0.1655, 0.2904, 0.4334, 0.5945, 0.7621, 0.9163, 1.0263, 1.0622, 1.0026, 0.8544, 0.6424, 0.4479, 0.2835, 0.1649, 0.0874, 0.0468, 0.0227, 0.0114])
    cmf_y = np.array([0.0004, 0.0012, 0.0040, 0.0116, 0.0230, 0.0380, 0.0600, 0.0910, 0.1390, 0.2080, 0.3230, 0.5030, 0.7100, 0.8620, 0.9540, 0.9950, 0.9950, 0.9520, 0.8700, 0.7570, 0.6310, 0.5030, 0.3810, 0.2650, 0.1750, 0.1070, 0.0610, 0.0320, 0.0170, 0.0082, 0.0041])
    cmf_z = np.array([0.0679, 0.2074, 0.6456, 1.3856, 1.7471, 1.7721, 1.6692, 1.2876, 0.8130, 0.4652, 0.2720, 0.1582, 0.0782, 0.0422, 0.0203, 0.0087, 0.0039, 0.0021, 0.0017, 0.0011, 0.0008, 0.0003, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    d65 = np.array([82.75, 91.49, 93.43, 86.68, 104.86, 117.01, 117.81, 114.86, 115.92, 108.81, 109.35, 107.80, 104.79, 107.69, 104.41, 104.05, 100.00, 96.33, 95.79, 88.69, 90.01, 89.60, 87.70, 83.29, 83.70, 80.03, 80.21, 82.28, 78.28, 69.72, 71.61])

    N = np.sum(d65 * cmf_y)
    X = np.sum(R * d65 * cmf_x) / N
    Y = np.sum(R * d65 * cmf_y) / N
    Z = np.sum(R * d65 * cmf_z) / N

    xyz_to_rgb = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
    rgb_linear = np.clip(xyz_to_rgb.dot(np.array([X, Y, Z])), 0, 1)

    rgb = np.where(rgb_linear <= 0.0031308, 12.92 * rgb_linear, 1.055 * (rgb_linear**(1/2.4)) - 0.055)
    rgb_255 = np.clip(np.round(rgb * 255), 0, 255).astype(int)

    return '#{:02X}{:02X}{:02X}'.format(rgb_255[0], rgb_255[1], rgb_255[2])

# 5. QTX 파싱 함수
def parse_qtx(text):
    blocks = re.split(r"\[(?:STANDARD_DATA|BATCH_DATA)\]", text)
    rows = []
    for block in blocks:
        prefix = "STD" if "STD_R=" in block else "BAT" if "BAT_R=" in block else None
        if not prefix: continue
        r_match = re.search(fr"{prefix}_R=([0-9.,\s]+)", block)
        if r_match:
            r_vals = [float(x) for x in re.split(r'[,\s]+', r_match.group(1).strip()) if x]
            for i, r in enumerate(r_vals[:31]):
                rows.append({"Wavelength": 400 + i*10, "Target_KS": calc_ks([r])[0]})
    return pd.DataFrame(rows).groupby("Wavelength")["Target_KS"].mean().reset_index()

# 6. 전수조사 수학적 최적화 엔진
def find_best_concentration(dye_name, target_ks_array):
    dye_df = df[df['Dye_Name'] == dye_name].sort_values('Concentration')
    concs = dye_df['Concentration'].values
    ks_matrix = np.array([calc_ks([row[f"R_{w}"] for w in range(400, 701, 10)]) for _, row in dye_df.iterrows()])
    
    if len(concs) == 1:
        err = np.sqrt(np.mean((target_ks_array - ks_matrix[0])**2))
        return concs[0], ks_matrix[0], err, concs[0], ks_matrix[0]
        
    interpolator = interp1d(concs, ks_matrix, axis=0, kind='linear')
    
    test_concs = np.linspace(concs.min(), concs.max(), 1000)
    test_ks_matrix = interpolator(test_concs)
    
    errors = np.sqrt(np.mean((target_ks_array - test_ks_matrix)**2, axis=1))
    best_idx = np.argmin(errors)
    
    best_conc = test_concs[best_idx]
    best_pred_ks = test_ks_matrix[best_idx]
    min_error = errors[best_idx]
    
    db_best_idx = np.argmin(np.abs(concs - best_conc))
    db_closest_conc = concs[db_best_idx]
    db_closest_ks = ks_matrix[db_best_idx]
    
    return best_conc, best_pred_ks, min_error, db_closest_conc, db_closest_ks

# 7. UI 및 메인 로직
uploaded_file = st.file_uploader("분석할 QTX 파일 업로드", type=["qtx", "txt"])

if uploaded_file is not None:
    target_df = parse_qtx(uploaded_file.read().decode("utf-8", errors="ignore"))
    
    if not target_df.empty:
        target_ks = target_df.sort_values("Wavelength")["Target_KS"].values
        
        # --- 🏆 [변경] 1. 추천 염료 TOP 5 영역 최상단 배치 ---
        st.header("🏆 추천 염료 TOP 5 (종합 매칭 분석 결과)")
        ranking = []
        TOLERANCE = 0.15  
        
        for name in sorted(df['Dye_Name'].unique()):
            p_conc, p_ks, err, _, _ = find_best_concentration(name, target_ks)
            
            t_flat, p_flat = target_ks.flatten(), p_ks.flatten()
            shape_sim = 0 if np.std(t_flat) == 0 or np.std(p_flat) == 0 else np.corrcoef(t_flat, p_flat)[0, 1]
            
            status = "합격" if err <= TOLERANCE else "불합격"
            match_score = (1 / (1 + err)) * shape_sim 

            ranking.append({
                "염료명": name, 
                "최적 농도(%)": p_conc, 
                "오차(RMSE)": err, 
                "형태 유사도": shape_sim,
                "매칭 점수": match_score,
                "판정": status
            })
        
        rank_df = pd.DataFrame(ranking).sort_values("매칭 점수", ascending=False).head(5)
        rank_df.insert(0, '순위', [f"{i+1}위" for i in range(len(rank_df))])
        
        def highlight_status(row):
            return ['background-color: #d4edda' if row['판정'] == '합격' else 'background-color: #f8d7da' for _ in row]

        display_df = rank_df[["순위", "염료명", "최적 농도(%)", "오차(RMSE)", "형태 유사도", "판정"]]
        st.dataframe(display_df.style.apply(highlight_status, axis=1).hide(axis="index"), use_container_width=True)
        
        # 💡 스마트 연동: 1등으로 뽑힌 염료의 이름을 기본값으로 추출
        top1_dye = rank_df.iloc[0]["염료명"]
        
        # --- 🔍 [변경] 2. 타이핑 검색 기능 탑재 상세 분석 영역 ---
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.header("🔍 특정 염료 상세 분석 및 시각화 비교")
        
        # 사이드바에서 메인 화면으로 이동하고, 검색 창 형태로 자동 매칭 수행
        all_dyes = sorted(df['Dye_Name'].unique())
        default_index = all_dyes.index(top1_dye) if top1_dye in all_dyes else 0
        
        selected_dye = st.selectbox(
            "상세하게 확인하고 싶은 염료명을 직접 입력하여 검색하거나 선택하세요:", 
            all_dyes, 
            index=default_index
        )
        
        # 사용자가 선택(혹은 검색)한 염료 데이터 기반 실시간 재연산
        pred_conc, pred_ks, error, closest_conc, closest_ks = find_best_concentration(selected_dye, target_ks)
        
        # 분석 결과 화면 출력
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 최적 매칭 농도", f"{pred_conc:.4f} %")
        c2.metric("📉 최소 오차율 (RMSE)", f"{error:.4f}")
        c3.metric("🧪 선택 염료", selected_dye)

        st.subheader("🎨 시각적 색상 매칭 확인")
        target_color = ks_to_hex(target_ks)
        pred_color = ks_to_hex(pred_ks)
        db_color = ks_to_hex(closest_ks)
        
        st.markdown(f"""
        <div style="display: flex; gap: 30px; margin-bottom: 30px;">
            <div style="text-align: center;">
                <div style="width: 100px; height: 100px; background-color: {target_color}; border-radius: 15px; border: 2px solid #ddd; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"></div>
                <p style="margin-top: 10px; font-weight: bold; font-size: 14px;">🎯 목표 색상 (Target)</p>
            </div>
            <div style="text-align: center;">
                <div style="width: 100px; height: 100px; background-color: {pred_color}; border-radius: 15px; border: 2px solid #ddd; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"></div>
                <p style="margin-top: 10px; font-weight: bold; font-size: 14px;">✨ 최적 처방<br>({pred_conc:.4f}%)</p>
            </div>
            <div style="text-align: center;">
                <div style="width: 100px; height: 100px; background-color: {db_color}; border-radius: 15px; border: 2px solid #ddd; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"></div>
                <p style="margin-top: 10px; font-weight: bold; font-size: 14px;">📂 DB 샘플<br>({closest_conc}%)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📊 매칭 결과 K/S 곡선 비교")
        chart_data = pd.DataFrame(index=range(400, 701, 10))
        chart_data["🎯 목표 색상 (Target)"] = target_ks
        chart_data[f"✨ 최적 처방 ({pred_conc:.4f}%)"] = pred_ks
        chart_data[f"📂 DB 샘플 ({closest_conc}%)"] = closest_ks
       
        st.line_chart(
            chart_data,
            color=["#EB2A2A", "#AC46EB", "#2A5DEB"] 
        )

    else:
        st.error("데이터를 분석할 수 없습니다.")
else:
    st.info("QTX 파일을 업로드하면 최적 농도값이 계산됩니다.")