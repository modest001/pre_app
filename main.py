import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import lime
from PIL import Image


@st.fragment
def para_input(model, explainer, explainer2, ct):
    st.header("请在下方输入相应指标👇", anchor=False)
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
    
    direction_dict = {"平面": 0, "北": 1, "东北": 2, "东": 3, "东南": 4, "南": 5, "西南": 6, "西": 7, "西北": 8}
    slope = st.selectbox("坡向(Aspect):", ["平面", "北", "东北", "东", "东南", "南", "西南", "西", "西北"])
    f11 = direction_dict[slope]

    f12 = st.slider("地形湿度指数(TWI):", min_value=-1.18, max_value=35.08, value=-1.18, step=0.01, format="%0.2f")
    
    f13 = st.slider("到最近铁路距离(Dis_to_road):", min_value=0, max_value=664042, value=0, step=1)
    f14 = st.slider("到最近道路距离(Dis_to_railway):", min_value=0.0, max_value=23136.3, value=0.0, step=0.1, format="%0.1f")
    f15 = st.slider("到最近居民点距离(Dis_to_sett):", min_value=0.0, max_value=25385.9, value=0.0, step=0.1, format="%0.1f")
    f16 = st.slider("人口密度(Den_pop):", min_value=0.688, max_value=15025.000, value=0.688, step=0.001, format="%0.3f")
    f17 = st.slider("人均GDP(GDP):", min_value=0.275, max_value=109917.000, value=0.275, step=0.001, format="%0.3f")
    
    forest_dict = {"针叶林": 0, "针阔叶混交林": 1, "阔叶林": 2, "灌丛": 3,
                    "草丛": 4, "草甸": 5, "高山植被": 6, "栽培植被": 7, "其他": 8}
    forest = st.selectbox("植被类型(Forest):", ["针叶林", "针阔叶混交林", "阔叶林", "灌丛", "草丛", "草甸", "高山植被", "栽培植被", "其他"])
    f18 = forest_dict[forest]

    data = {'Da_AVGTEM': [f1], 'Da_PRE': [f2], 'Da_AVGRH': [f3], 'Da_AVGWIN': [f4], 'Da_AVGPRS': [f5], 'SSD': [f6], 'Da_MAXWIN': [f7], 
            'Da_MAXGST': [f8], 'Elevation': [f9], 'Slope': [f10], 'Aspect': [f11], 'TWI': [f12], 'Dis_to_railway': [f13], 
            'Dis_to_road': [f14], 'Dis_to_sett': [f15], 'Den_pop': [f16], 'GDP': [f17], 'Forest': [f18]}
    features = pd.DataFrame(data, columns=feature_names)
    st.session_state["features"] = features
    pre_button = st.button('预测', type='primary')
    if pre_button:
        with ct:
            main(model, explainer, explainer2)


@st.fragment
def main(model, explainer, explainer2):
    feature_names = ['Da_AVGTEM', 'Da_PRE', 'Da_AVGRH', 'Da_AVGWIN', 'Da_AVGPRS', 'SSD', 'Da_MAXWIN', 
                    'Da_MAXGST', 'Elevation', 'Slope', 'Aspect', 'TWI', 'Dis_to_railway', 'Dis_to_road', 
                    'Dis_to_sett', 'Den_pop', 'GDP', 'Forest']
    if True:
        # 添加主标题和模型信息
        st.title("四川和云南省森林火灾风险预测")
        
        # 关于模型部分
        st.subheader("关于模型")
        st.write("该模型的内部验证结果显示,其ROC曲线下面积(AUC)为 0.908 (95%CI: 0.82-0.933)，表明该模型具有很强的预测性能，校准曲线和决策曲线分析表明该模型具有预测森林火灾的能力。")
        
        # 计算器指南
        st.subheader("网页计算器指南")
        st.write("计算器由三个主要部分组成:第一部分的左侧边栏允许用户输入相关参数并选择模型变量，第二部分显示院内森林火灾的预测概率。第三部分提供了详细的模型信息，包括使用SHAP和LIME进行的局部解释，为预测结果提供解释。希望本指南能帮助您有效利用我们的预测计算器。")

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
        st.write("SHAP瀑布图中红色代表该因子对模型预测有正向贡献，蓝色代表该因子对模型预测有负向贡献，同时，因子的颜色条长度越大，代表该样本的特征取值对预测结果的影响越大。")

        # LIME解释
        st.subheader("LIME局部解释", anchor=False)
        exp2 = explainer2.explain_instance(
            data_row=st.session_state["features"].values[0],
            predict_fn=model.predict_proba,
            num_features=11
        )
        
        fig2 = exp2.as_pyplot_figure()
        st.pyplot(fig2)
        st.write("LIME图中绿色代表该因子对预测有正向贡献，红色代表该因子对预测有负向贡献。")


# 修改后的初始化部分
if __name__ == "__main__":
    # 固定全局随机种子
    import numpy as np
    import random
    np.random.seed(42)
    random.seed(42)

    model = joblib.load('lgbml.pkl')
    explainer = shap.TreeExplainer(model)

    data1 = pd.read_excel('./数据删.xls')
    columns_to_drop = ['LONGITUDE','LATITUDE','火点','TMX','TMN','GST']
    X = data1.drop(columns=columns_to_drop)
    X.rename(columns={...}, inplace=True)  # 保持原有重命名逻辑

    # 预处理训练数据
    X_processed = ct.transform(X)

    # 获取分类变量索引（假设Aspect和Forest在预处理后的位置为10和17）
    categorical_features = [10, 17]

    # 初始化LIME解释器
    explainer2 = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_processed,
        feature_names=ct.get_feature_names_out(),
        categorical_features=categorical_features,
        discretize_continuous=False,
        class_names=['No Fire', 'Fire'],
        mode='classification',
        random_state=42
    )

# 修改后的预测函数
def predict_fn(x):
    x_df = pd.DataFrame(x, columns=ct.get_feature_names_out())
    return model.predict_proba(x_df)

# 修改后的main函数
def main(model, explainer, explainer2):
    features_processed = ct.transform(st.session_state["features"])
    exp2 = explainer2.explain_instance(
        data_row=features_processed[0],
        predict_fn=predict_fn,
        num_features=11
    )
    fig2 = exp2.as_pyplot_figure()
    st.pyplot(fig2)
