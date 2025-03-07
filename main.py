import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import lime
from PIL import Image

# 自定义CSS样式（放在最前面）
st.markdown("""
<style>
/* 主标题样式 */
.custom-main-title {
    font-size: 36px !important;
    color: #2E86C1;
    font-weight: bold;
    text-align: center;
    margin-bottom: 30px;
}

/* 二级标题样式（侧边栏标题和章节标题） */
.custom-sub-title {
    font-size: 28px !important;
    color: #2ECC71;
    border-bottom: 2px solid #F4D03F;
    padding-bottom: 5px;
    margin-top: 25px !important;
}

/* 内容文本样式 */
.content-text {
    font-size: 16px !important;  /* 与滑块字体一致 */
    line-height: 1.6;
}

/* 图表容器样式 */
.plot-container {
    border: 1px solid #D5D8DC;
    border-radius: 8px;
    padding: 15px;
    margin: 15px 0;
}
</style>
""", unsafe_allow_html=True)
@st.fragment
def para_input(model, explainer, explainer2, ct):
    # 侧边栏标题应用样式
    st.markdown('<div class="custom-sub-title">请在下方输入相应指标👇</div>', unsafe_allow_html=True)
    feature_names = ['Da_AVGTEM', 'Da_PRE', 'Da_AVGRH', 'Da_AVGWIN', 'Da_AVGPRS', 'SSD', 'Da_MAXWIN', 
                    'Da_MAXGST', 'Elevation', 'Slope', 'Aspect', 'TWI', 'Dis_to_railway', 'Dis_to_road', 
                    'Dis_to_sett', 'Den_pop', 'GDP', 'Forest']
    f1 = st.slider("日平均温度(Da_AVGTEM):", min_value=-17, max_value=40, value=-17, step=1)
    f2 = st.slider("日降雨量(Da_PRE):", min_value=0, max_value=99, value=0, step=1)
    f3 = st.slider("日平均相对湿度(Da_AVGRH):", min_value=0, max_value=100, value=0, step=1)
    f4 = st.slider("日平均风速(Da_AVGWIN):", min_value=0.0, max_value=9.3, value=0.0, step=0.1, format="%0.1f")
    f5 = st.slider("日平均气压(Da_AVGPRS):", min_value=602.0, max_value=1008.7, value=602.0, step=0.1, format="%0.1f")
    f6 = st.slider("日照时数(SSD):", min_value=0.0, max_value=13.5, value=0.0, step=0.1, format="%0.1f")

    f7 = st.slider("日最大风速(Da_MAXWIN):", min_value=0, max_value=25, value=0, step=1)
    f8 = st.slider("日最高地表气温(Da_MAXGST):", min_value=-2.9, max_value=78.6, value=-2.9, step=0.1, format="%0.1f")
    f9 = st.slider("海拔(Elevation):", min_value=1, max_value=7713, value=1, step=1)
    f10 = st.slider("坡度(Slope):", min_value=0.0, max_value=89.3, value=0.0, step=0.1, format="%0.1f")
    # f11 = st.slider("坡向/Aspect:", min_value=1, max_value=8, value=1, step=1)
    direction_dict = {"平面": 0, "北": 1, "东北": 2, "东": 3, "东南": 4, "南": 5, "西南": 6, "西": 7, "西北": 8}
    slope = st.selectbox("坡向(Aspect):", ["平面", "北", "东北", "东", "东南", "南", "西南", "西", "西北"])
    f11 = direction_dict[slope]

    f12 = st.slider("地形湿度指数(TWI):", min_value=-1.18, max_value=35.08, value=-1.18, step=0.01, format="%0.2f")
    
    f13 = st.slider("到最近铁路距离(Dis_to_road):", min_value=0, max_value=664042, value=0, step=1)
    f14 = st.slider("到最近道路距离(Dis_to_railway):", min_value=0.0, max_value=23136.3, value=0.0, step=0.1, format="%0.1f")
    f15 = st.slider("到最近居民点距离(Dis_to_sett):", min_value=0.0, max_value=25385.9, value=0.0, step=0.1, format="%0.1f")
    f16 = st.slider("人口密度(Den_pop):", min_value=0.688, max_value=15025.000, value=0.688, step=0.001, format="%0.3f")
    f17 = st.slider("人均GDP(GDP):", min_value=0.275, max_value=109917.000, value=0.275, step=0.001, format="%0.3f")
    # f18 = st.slider("Forest:", min_value=0, max_value=8, value=0, step=1)
    forest_dict = {"针叶林": 0, "针阔叶混交林": 1, "阔叶林": 2, "灌丛": 3,
                    "草丛": 4, "草甸": 5, "高山植被": 6, "栽培植被": 7, "其他": 8}
    forest = st.selectbox("植被类型(Forest):", ["针叶林", "针阔叶混交林", "阔叶林", "灌丛", "草丛", "草甸", "高山植被", "栽培植被", "其他"])
    f18 = forest_dict[forest]

    data = {'Da_AVGTEM': [f1], 'Da_PRE': [f2], 'Da_AVGRH': [f3], 'Da_AVGWIN': [f4], 'Da_AVGPRS': [f5], 'SSD': [f6], 'Da_MAXWIN': [f7], 
            'Da_MAXGST': [f8], 'Elevation': [f9], 'Slope': [f10], 'Aspect': [f11], 'TWI': [f12], 'Dis_to_railway': [f13], 
            'Dis_to_road': [f14], 'Dis_to_sett': [f15], 'Den_pop': [f16], 'GDP': [f17], 'Forest': [f18]}
    features = pd.DataFrame(data, columns=feature_names)
    if st.button('预测', type='primary', key='predict_btn'):
        st.session_state["show_results"] = True
    else:
        # 确保未点击时不显示结果
        st.session_state["show_results"] = False


@st.fragment
def main(model, explainer, explainer2):
    # 主标题
    st.markdown('<div class="custom-main-title">四川云南省森林火灾预测系统</div>', 
               unsafe_allow_html=True)
    
    # 关于模型章节
    st.markdown('<div class="custom-sub-title">一. 关于模型</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="content-text">
    - 模型AUC：0.962（ROC曲线下面积）<br>
    - 算法：LightGBM集成学习<br>
    - 输入特征：气象、地形、人文等18个指标<br>
    - 训练数据：四川云南历史火灾数据
    </div>
    """, unsafe_allow_html=True)
    
    # 实时预测章节
    st.markdown('<div class="custom-sub-title">二. 实时预测</div>', unsafe_allow_html=True)
    
    # 结果展示容器
    results_container = st.container()
    
    # 严格的结果显示控制
    if st.session_state.get("show_results", False):
        with results_container:
            try:
                # 预测执行
                features = st.session_state["features"]
                fire_type = model.predict(features)
                predicted_proba = model.predict_proba(features)[0]
                types = ["不发生火灾", "发生火灾"]
                
                # 结果展示
                st.success(f'<div class="content-text">预测结果：{types[fire_type[0]]}（概率：{predicted_proba[fire_type[0]]:.2f}）</div>', 
                          unsafe_allow_html=True)
                
                # 解释章节
                st.markdown('<div class="custom-sub-title">三. SHAP和LIME局部解释</div>', 
                           unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                
                # SHAP解释
                with col1:
                    with st.spinner("生成SHAP解释..."):
                        shap_values = explainer.shap_values(features)
                        exp = shap.Explanation(shap_values, explainer.expected_value, 
                                             features, feature_names=features.columns)
                        fig1 = plt.figure(figsize=(10, 6))
                        shap.waterfall_plot(exp[0], max_display=11)
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        st.pyplot(fig1)
                        st.markdown('</div>', unsafe_allow_html=True)
                        plt.close()
                        st.markdown("""
                        <div class="content-text">
                        SHAP值显示各特征对预测结果的贡献方向（正/负）和强度：<br>
                        - 基准值（base value）：模型平均预测值<br>
                        - 最终值（f(x)）：当前样本预测值
                        </div>
                        """, unsafe_allow_html=True)
                
                # LIME解释
                with col2:
                    with st.spinner("生成LIME解释..."):
                        exp2 = explainer2.explain_instance(
                            features.values[0], 
                            model.predict_proba, 
                            num_features=11
                        )
                        fig2 = exp2.as_pyplot_figure()
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        st.pyplot(fig2)
                        st.markdown('</div>', unsafe_allow_html=True)
                        plt.close()
                        st.markdown("""
                        <div class="content-text">
                        LIME局部解释说明：<br>
                        - 绿色特征：促进火灾预测的因素<br>
                        - 红色特征：抑制火灾预测的因素<br>
                        - 数值大小：特征重要性程度
                        </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"发生错误：{str(e)}")
                st.session_state["show_results"] = False

if __name__ == "__main__":
    # 初始化会话状态
    if "show_results" not in st.session_state:
        st.session_state["show_results"] = False
    
    # 缓存模型加载（解决界面变灰问题）
    @st.cache_resource
    def load_resources():
        model = joblib.load('lgbml.pkl')
        data1 = pd.read_excel('./数据删.xls')
        # 数据预处理...
        explainer = shap.TreeExplainer(model)
        explainer2 = lime.lime_tabular.LimeTabularExplainer(...)
        return model, explainer, explainer2
    
    model, explainer, explainer2 = load_resources()
    
    # 页面布局
    main_container = st.container()
    with main_container:
        main(model, explainer, explainer2)
    
    with st.sidebar:
        para_input(model, explainer, explainer2, main_container)
