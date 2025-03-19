import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import lime
from PIL import Image

# 静态内容在页面加载时显示
st.title("四川和云南省森林火灾风险预测")
st.subheader("关于模型")
st.write(
    "该模型的内部验证结果显示,其ROC曲线下面积(AUC)为 0.908 (95%CI: 0.82-0.933)，表明该模型具有很强的预测性能，校准曲线和决策曲线分析表明该模型具有预测森林火灾的能力。"
)
st.subheader("网页计算器指南")
st.write(
    "计算器由三个主要部分组成:第一部分的左侧边栏允许用户输入相关参数并选择模型变量，第二部分显示院内森林火灾的预测概率。第三部分提供了详细的模型信息，包括使用SHAP和LIME进行的局部解释，为预测结果提供解释。希望本指南能帮助您有效利用我们的预测计算器。"
)

@st.fragment
def para_input(model, explainer, explainer2, ct):
    st.header("请在下方输入相应指标👇", anchor=False)
    feature_names = ['Da_AVGTEM', 'Da_PRE', 'Da_AVGRH', 'Da_AVGWIN', 'Da_AVGPRS', 'SSD', 'Da_MAXWIN',
                     'Da_MAXGST', 'Elevation', 'Slope', 'Aspect', 'TWI', 'Dis_to_railway', 'Dis_to_road',
                     'Dis_to_sett', 'Den_pop', 'GDP', 'Forest']
    # ... [保留原有的滑动条和选择框代码] ...

    data = {'Da_AVGTEM': [f1], 'Da_PRE': [f2], 'Da_AVGRH': [f3], 'Da_AVGWIN': [f4], 'Da_AVGPRS': [f5], 'SSD': [f6],
            'Da_MAXWIN': [f7],
            'Da_MAXGST': [f8], 'Elevation': [f9], 'Slope': [f10], 'Aspect': [f11], 'TWI': [f12],
            'Dis_to_railway': [f13],
            'Dis_to_road': [f14], 'Dis_to_sett': [f15], 'Den_pop': [f16], 'GDP': [f17], 'Forest': [f18]}
    features = pd.DataFrame(data, columns=feature_names)
    st.session_state["features"] = features
    pre_button = st.button('预测', type='primary')
    if pre_button:
        with ct:
            main(model, explainer, explainer2)

@st.fragment
def main(model, explainer, explainer2):
    # 预测结果和解释部分
    feature_names = ['Da_AVGTEM', 'Da_PRE', 'Da_AVGRH', 'Da_AVGWIN', 'Da_AVGPRS', 'SSD', 'Da_MAXWIN',
                     'Da_MAXGST', 'Elevation', 'Slope', 'Aspect', 'TWI', 'Dis_to_railway', 'Dis_to_road',
                     'Dis_to_sett', 'Den_pop', 'GDP', 'Forest']
    if "features" in st.session_state and not st.session_state["features"].empty:
        # 预测结果
        fire_type = model.predict(st.session_state["features"])
        predicted_proba = model.predict_proba(st.session_state["features"])[0]
        types = ["不发生火灾", "发生火灾"]

        st.subheader("预测结果", anchor=False)
        st.write(f'预测结果为：{types[fire_type[0]]}，概率为{round(predicted_proba[fire_type[0]], 2)}。')

        # SHAP解释
        st.subheader("SHAP局部解释", anchor=False)
        shap_values = explainer.shap_values(st.session_state["features"])
        exp = shap.Explanation(shap_values, explainer.expected_value, st.session_state["features"],
                               feature_names=st.session_state["features"].columns)

        fig, _ = plt.subplots()
        shap.waterfall_plot(exp[0], max_display=11)
        st.pyplot(fig)
        st.write(
            "SHAP瀑布图中红色代表该因子对模型预测有正向贡献，蓝色代表该因子对模型预测有负向贡献，同时，因子的颜色条长度越大，代表该样本的特征取值对预测结果的影响越大。")

        # LIME解释
        explainer2 = lime.lime_tabular.LimeTabularExplainer(
            training_data=X.values,
            feature_names=X.columns.tolist(),
            class_names=['No Fire', 'Fire'],
            mode='classification',
            random_state=42
        )
        st.subheader("LIME局部解释", anchor=False)
        exp2 = explainer2.explain_instance(
            data_row=st.session_state["features"].values[0],
            predict_fn=model.predict_proba,
            num_features=11
        )

        fig2 = exp2.as_pyplot_figure()
        st.pyplot(fig2)
        st.write("LIME图中绿色代表该因子对预测有正向贡献，红色代表该因子对预测有负向贡献。")

if __name__ == "__main__":
    model = joblib.load('lgbml.pkl')
    explainer = shap.TreeExplainer(model)

    data1 = pd.read_excel('./数据删.xls')
    columns_to_drop = ['LONGITUDE', 'LATITUDE', '火点', 'TMX', 'TMN', 'GST']
    X = data1.drop(columns=columns_to_drop)
    X.rename(columns={'TEM': 'Da_AVGTEM', 'TMN': 'Da_MINTEM', 'TMX': 'Da_MAXTEM', 'PRE': 'Da_PRE',
                      'WIN': 'Da_AVGWIN', 'PRS': 'Da_AVGPRS', 'GST': 'Da_AVGGST', 'WINMAX': 'Da_MAXWIN',
                      'GSTMAX': 'Da_MAXGST', 'RHU': 'Da_AVGRH', '高程': 'Elevation', '坡度': 'Slope',
                      '坡向': 'Aspect', '铁路欧': 'Dis_to_railway', '公路欧': 'Dis_to_road',
                      '平均人': 'Den_pop', '平均gdp': 'GDP', '居民欧': 'Dis_to_sett', 'forest': 'Forest',
                      'twi': 'TWI'}, inplace=True)

    explainer2 = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=['No Fire', 'Fire'],
        mode='classification',
        random_state=42
    )
    if "features" not in st.session_state:
        st.session_state["features"] = pd.DataFrame()

    ct = st.container()
    with st.sidebar:
        para_input(model, explainer, explainer2, ct)
