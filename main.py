import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import lime
from PIL import Image

# ==================== 密码验证系统 ====================
st.set_page_config(
    page_title="森林火灾预测系统",
    page_icon="🔥",
    layout="wide"
)

# 初始化session状态
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'password_attempts' not in st.session_state:
    st.session_state.password_attempts = 0

# 设置密码（这里设置为123456）
CORRECT_PASSWORD = "123456"
MAX_ATTEMPTS = 5  # 最多尝试次数

def check_password():
    """密码验证函数"""
    
    def password_entered():
        """检查输入的密码"""
        entered_password = st.session_state["password_input"]
        
        if entered_password == CORRECT_PASSWORD:
            st.session_state.authenticated = True
            st.session_state.password_input = ""  # 清除密码
            st.session_state.password_attempts = 0  # 重置尝试次数
            st.rerun()
        else:
            st.session_state.password_attempts += 1
            st.error(f"❌ 密码错误！")
            
            # 显示剩余尝试次数
            remaining = MAX_ATTEMPTS - st.session_state.password_attempts
            if remaining > 0:
                st.warning(f"剩余尝试次数: {remaining}")
            else:
                st.error("⛔ 尝试次数过多，请稍后再试")
                st.stop()
    
    # 如果已经认证
    if st.session_state.authenticated:
        return True
    
    # 检查尝试次数
    if st.session_state.password_attempts >= MAX_ATTEMPTS:
        st.error("⛔ 尝试次数过多，系统已锁定")
        st.stop()
    
    # 显示登录界面
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🔥 森林火灾危险预测系统</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 创建两列布局
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # 系统信息卡片
        st.markdown("### 📋 系统信息")
        info_card = st.container(border=True)
        with info_card:
            st.markdown("**单位**: 南京工业大学")
            st.markdown("**作者**: 刘之扬")
            st.markdown("**版本**: 1.0")
            st.markdown("**功能**: 四川和云南省森林火灾危险预测")
        
        st.markdown("---")
        
        # 密码输入
        st.markdown("### 🔐 访问控制")
        st.text_input(
            "请输入访问密码",
            type="password",
            key="password_input",
            help="密码提示：123456",
            on_change=password_entered
        )
        
        # 提示信息
        st.caption("ℹ️ 请输入密码访问系统")
        st.caption("⚠️ 如需访问权限，请联系：刘之扬")
        
        # 调试信息（正式使用时可以删除）
        if st.checkbox("显示调试信息"):
            st.info(f"当前密码尝试次数: {st.session_state.password_attempts}")
            st.info(f"正确密码是: {CORRECT_PASSWORD}")
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>© 2024 南京工业大学 刘之扬</p>", unsafe_allow_html=True)
    
    st.stop()
    return False

# 调用密码检查
check_password()

# ==================== 主程序 ====================
# 只有密码正确才会执行下面的代码

# 显示当前用户信息
st.sidebar.success(f"✅ 已登录 | 单位：南京工业大学")

# 登出按钮
if st.sidebar.button("🚪 退出登录"):
    st.session_state.authenticated = False
    st.session_state.password_attempts = 0
    st.rerun()

# 主内容区
st.title("四川和云南省森林火灾危险预测")
st.markdown("**单位：南京工业大学 | 作者：刘之扬**")

st.subheader("关于模型")
st.write(
    "该模型的内部验证结果显示,其ROC曲线下面积(AUC)为 0.962，表明该模型具有很强的预测性能，森林火灾危险等级分析表明该模型能够有效的划分森林危险区。"
)

st.subheader("网页计算器指南")
st.write(
    "计算器由三个主要部分组成:第一部分的左侧边栏允许用户输入相关参数变量，第二部分显示对此样本森林火灾的预测概率。第三部分提供了详细的模型信息，包括使用SHAP和LIME进行的局部解释，为预测结果提供解释。希望本指南能帮助您有效利用我们的预测计算器。"
)

# 添加侧边栏的使用说明
with st.sidebar:
    st.header("📖 使用说明")
    st.info("""
    1. 在左侧输入相关参数
    2. 系统会自动计算火灾风险
    3. 查看下方的分析结果
    4. 使用完毕后点击退出登录
    """)

# ==================== 以下是原有的功能代码 ====================
# 你需要在这里添加原来的预测功能代码

# 示例：创建一个简单的输入表单
st.sidebar.header("📊 输入参数")

# 这里添加原有的参数输入代码
# 例如：
# temperature = st.sidebar.slider("温度", 0, 50, 25)
# humidity = st.sidebar.slider("湿度", 0, 100, 50)

st.sidebar.markdown("---")
st.sidebar.markdown("**⚠️ 安全提示**: 使用完毕后请及时退出登录")

# 主界面继续原有的功能
st.header("📈 预测结果")

# 这里添加原有的结果显示代码

st.markdown("---")
st.markdown("### 📞 联系方式")
st.write("**单位**: 南京工业大学")
st.write("**作者**: 刘之扬")
st.write("**邮箱**: [请填写你的邮箱]")
st.write("**电话**: [请填写你的电话]")

# 页脚
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>南京工业大学 森林火灾预测系统 v1.0 © 2024</p>", unsafe_allow_html=True)

@st.fragment
def para_input(model, explainer, explainer2, ct):
    st.header("请在下方输入相应指标👇", anchor=False)
    feature_names = ['Da_AVGTEM', 'Da_PRE', 'Da_AVGRH', 'Da_AVGWIN', 'Da_AVGPRS', 'SSD', 'Da_MAXWIN',
                     'Da_MAXGST', 'Elevation', 'Slope', 'Aspect', 'TWI', 'Dis_to_railway', 'Dis_to_road',
                     'Dis_to_sett', 'Den_pop', 'GDP', 'Forest']
    
    # 确保所有输入控件的标签使用英文标点
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
    
    # 修复特征名与变量对应的错误
    f13 = st.slider("到最近铁路距离(Dis_to_railway):", min_value=0, max_value=664042, value=0, step=1)
    f14 = st.slider("到最近道路距离(Dis_to_road):", min_value=0.0, max_value=23136.3, value=0.0, step=0.1, format="%0.1f")
    f15 = st.slider("到最近居民点距离(Dis_to_sett):", min_value=0.0, max_value=25385.9, value=0.0, step=0.1, format="%0.1f")
    f16 = st.slider("人口密度(Den_pop):", min_value=0.688, max_value=15025.000, value=0.688, step=0.001, format="%0.3f")
    f17 = st.slider("人均GDP(GDP):", min_value=0.275, max_value=109917.000, value=0.275, step=0.001, format="%0.3f")

    forest_dict = {"针叶林": 0, "针阔叶混交林": 1, "阔叶林": 2, "灌丛": 3,
                   "草丛": 4, "草甸": 5, "高山植被": 6, "栽培植被": 7, "其他": 8}
    forest = st.selectbox("植被类型(Forest):", ["针叶林", "针阔叶混交林", "阔叶林", "灌丛", "草丛", "草甸", "高山植被", "栽培植被", "其他"])
    f18 = forest_dict[forest]

    # 确保所有变量名正确引用
    data = {
        'Da_AVGTEM': [f1], 'Da_PRE': [f2], 'Da_AVGRH': [f3], 
        'Da_AVGWIN': [f4], 'Da_AVGPRS': [f5], 'SSD': [f6],
        'Da_MAXWIN': [f7], 'Da_MAXGST': [f8], 'Elevation': [f9], 
        'Slope': [f10], 'Aspect': [f11], 'TWI': [f12],
        'Dis_to_railway': [f13], 'Dis_to_road': [f14], 
        'Dis_to_sett': [f15], 'Den_pop': [f16], 
        'GDP': [f17], 'Forest': [f18]
    }
    features = pd.DataFrame(data, columns=feature_names)
    st.session_state["features"] = features
    pre_button = st.button('预测', type='primary')
    if pre_button:
        with ct:
            main(model, explainer, explainer2)

@st.fragment
def main(model, explainer, explainer2):
    if "features" in st.session_state and not st.session_state["features"].empty:
        # 预测逻辑
        fire_type = model.predict(st.session_state["features"])
        predicted_proba = model.predict_proba(st.session_state["features"])[0]
        types = ["不发生火灾", "发生火灾"]
        st.subheader("预测结果", anchor=False)
        st.write(f'预测结果为：{types[fire_type[0]]}，概率为{round(predicted_proba[fire_type[0]], 2)}。')
        
        st.subheader("模型解释", anchor=False)
        col1, col2 = st.columns(2)  # 创建并排的两列
        
        with col1:
            st.markdown("**SHAP局部解释**")
            shap_values = explainer.shap_values(st.session_state["features"])
            exp = shap.Explanation(
                shap_values, 
                explainer.expected_value, 
                st.session_state["features"],
                feature_names=st.session_state["features"].columns
            )
            fig, _ = plt.subplots(figsize=(8,5))  # 调整图形大小
            shap.waterfall_plot(exp[0], max_display=11)  # 减少显示的特征数量
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("上图显示了SHAP瀑布图，其中红色代表该因子对模型有正向贡献，蓝色代表该因子对模型有负向贡献，长度表示增加（正值）或减少（负值）相对于基线的预测。")

        # LIME解释
        with col2:
            st.markdown("**LIME局部解释**")
            explainer2 = lime.lime_tabular.LimeTabularExplainer(
                     training_data=X.values,
                     feature_names=X.columns.tolist(),
                     class_names=['No Fire', 'Fire'],
                     mode='classification',
                     random_state=42
            )
            exp2 = explainer2.explain_instance(
                st.session_state["features"].values[0], 
                model.predict_proba, 
                num_features=11  # 减少显示的特征数量
            )
            fig2 = exp2.as_pyplot_figure()
            fig2.set_size_inches(8,5)  # 调整图形大小
            plt.tight_layout()
            st.pyplot(fig2)
            st.caption("上图显示了LIME图，绿色代表该因子对预测有正向贡献，红色代表该因子对预测有负向贡献。")

if __name__ == "__main__":
    model = joblib.load('lgbml.pkl')
    data1 = pd.read_excel('./数据删.xls')
    columns_to_drop = ['LONGITUDE', 'LATITUDE', '火点', 'TMX', 'TMN', 'GST']
    X = data1.drop(columns=columns_to_drop).rename(columns={
        'TEM': 'Da_AVGTEM', 'PRE': 'Da_PRE', 'WIN': 'Da_AVGWIN',
        'PRS': 'Da_AVGPRS', 'WINMAX': 'Da_MAXWIN', 'GSTMAX': 'Da_MAXGST',
        'RHU': 'Da_AVGRH', '高程': 'Elevation', '坡度': 'Slope',
        '坡向': 'Aspect', '铁路欧': 'Dis_to_railway', '公路欧': 'Dis_to_road',
        '居民欧': 'Dis_to_sett', '平均人': 'Den_pop', '平均gdp': 'GDP',
        'forest': 'Forest', 'twi': 'TWI'
    })
    explainer = shap.TreeExplainer(model)
    explainer2 = lime.lime_tabular.LimeTabularExplainer(
        X.values, 
        feature_names=X.columns.tolist(),
        class_names=['No Fire', 'Fire'],
        mode='classification',
        random_state=42
    )
    st.session_state.setdefault("features", pd.DataFrame())
    ct = st.container()
    with st.sidebar:
        para_input(model, explainer, explainer2, ct)


