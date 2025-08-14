import c2qa as c2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import qiskit
from qiskit.quantum_info.operators import Pauli
from qiskit import ClassicalRegister, AncillaRegister
from qiskit.visualization import plot_histogram
from qiskit_aer.noise import NoiseModel, phase_amplitude_damping_error,PauliLindbladError
import qiskit.visualization
from math import pi
# 设置支持中文的字体
matplotlib.rcParams["font.family"] = ["SimHei", "Times New Roman"]
# 解决负号显示问题（可选）
matplotlib.rcParams["axes.unicode_minus"] = False  # 正确显示负号
def prepare_cat_state(alpha, cutoff, is_even=True):
    """
    使用c2qa制备偶猫态或奇猫态

    参数:
        alpha: 相干态参数（平均光子数 = |alpha|²）
        cutoff: 截断维度（Fock态最大光子数）
        is_even: 若为True，制备偶猫态（|α⟩+|-α⟩）；否则制备奇猫态（|α⟩-|-α⟩）

    返回:
        qc: 包含猫态初始化的量子电路
        cat_state: 猫态的状态向量
    """
    # 1. 创建量子寄存器（1个qumode，用于表示连续变量猫态）
    qumodes = c2.QumodeRegister(1,3, name="q")
    measure_cat = ClassicalRegister(3)
    qc = c2.CVCircuit(qumodes,measure_cat)  # c2qa的连续变量电路

    # 2. 生成相干态 |α⟩
    # 创建临时电路制备|α⟩
    temp_qc_alpha = c2.CVCircuit(qumodes)
    temp_qc_alpha.cv_initialize(0, qumodes[0])  # 初始化为真空态 |0⟩
    temp_qc_alpha.cv_d(alpha,  qumodes[0])  # 应用位移门得到 |α⟩
    # 模拟得到|α⟩的状态向量
    state_alpha, _, _ = c2.util.simulate(temp_qc_alpha, shots=1)
    state_alpha = np.array(state_alpha, dtype=np.complex128)

    # 3. 生成相干态 |-α⟩
    temp_qc_minus_alpha = c2.CVCircuit(qumodes)
    temp_qc_minus_alpha.cv_initialize(0, qumodes[0])  # 真空态初始化
    temp_qc_minus_alpha.cv_d(-alpha,  qumodes[0])  # 位移门得到 |-α⟩
    state_minus_alpha, _, _ = c2.util.simulate(temp_qc_minus_alpha, shots=1, )
    state_minus_alpha = np.array(state_minus_alpha, dtype=np.complex128)

    # 4. 截断并对齐两个态的长度（确保维度一致）
    state_alpha = state_alpha[:cutoff]
    state_minus_alpha = state_minus_alpha[:cutoff]
    # 若长度不足，补零
    if len(state_alpha) < cutoff:
        state_alpha = np.pad(state_alpha, (0, cutoff - len(state_alpha)), mode='constant')
    if len(state_minus_alpha) < cutoff:
        state_minus_alpha = np.pad(state_minus_alpha, (0, cutoff - len(state_minus_alpha)), mode='constant')

    # 5. 构建猫态并归一化
    if is_even:
        cat_state = state_alpha + state_minus_alpha  # 偶猫态
    else:
        cat_state = state_alpha - state_minus_alpha  # 奇猫态
    cat_norm = np.linalg.norm(cat_state)
    if cat_norm < 1e-10:
        raise ValueError("猫态范数过小，可能是alpha或cutoff设置不当")
    cat_state /= cat_norm  # 归一化
    qc.reset(qumodes[0])
    qc.cv_initialize(cat_state, qumodes[0])
    qc.cv_measure( qumodes[0],measure_cat )
    _,result,counts=c2.util.simulate(qc)
    print(counts)
    # 绘制Wigner函数
    c2.wigner.plot_wigner(
        qc,  # 包含猫态的电路
        cat_state
    )

    return  cat_state

def plot_photon_distribution(state, title, cutoff):
    """绘制光子数分布"""
    photon_counts = np.abs(state[:cutoff])** 2  # 各光子数的概率
    plt.bar(range(cutoff), photon_counts)
    plt.xlabel("光子数 n")
    plt.ylabel("概率 |cₙ|²")
    plt.title(title)
    plt.show()
# --------------------------
# 双光子耗散：通过噪声模型实现
# --------------------------
def create_two_photon_dissipator(cutoff):
    """构造双光子耗散算符 L = a²（用于噪声模型）"""
    a = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for n in range(1, cutoff):
        a[n - 1, n] = np.sqrt(n)  # 湮灭算符 a
    a_squared = a @ a  # 双光子湮灭算符 a²
    return a_squared


def create_two_photon_noise_model(cutoff, gamma=0.1):
    """使用PauliLindbladError创建双光子耗散噪声模型"""
    # 对于连续变量系统，我们可以用等效的泡利算符组合来近似双光子耗散
    # 这里使用X和Y泡利算符的组合来模拟耗散效应
    X = Pauli('X')
    Y = Pauli('Y')

    # 创建PauliLindbladError，参数为(泡利算符列表, 噪声强度列表)
    # 这是一种近似方法，用泡利算符组合模拟双光子耗散
    pauli_error = PauliLindbladError(
        [X, Y],  # 泡利算符列表
        [gamma / 2, gamma / 2]  # 对应的噪声强度，总和为gamma
    )

    noise_model = NoiseModel()
    # 将噪声关联到连续变量门
    noise_model.add_all_qubit_quantum_error(pauli_error, ["cv_d", "cv_c_rx", "cv_measure"])
    return noise_model

# 使用cv_c_rx门实现受控操作
# --------------------------
def noise_biased_cx(qc, ancilla, data_qumode):
    """使用cv_c_rx门实现受控色散相互作用"""
    # 应用受控RX旋转门： qubit控制，qumode被控制
    qc.cv_c_rx(pi, [data_qumode], ancilla)

# --------------------------
# 稳定器测量
# --------------------------
def measure_stabilizer(qc, data_idx, ancilla_idx,cutoff,temp_reg):
    """测量重复码稳定器"""
    ancilla = ancillas[ancilla_idx]
    qc.h(ancilla)
    noise_biased_cx(qc, ancilla, qumodes[data_idx])
    noise_biased_cx(qc, ancilla, qumodes[data_idx + 1])
    qc.h(ancilla)
    qc.measure(ancilla, temp_reg[ancilla_idx])
    qc.reset(ancilla)
    return qc

# --------------------------
#  参数设置
# --------------------------

num_data = 5  # 数据猫量子比特数量
num_ancilla = 4  # 辅助transmon数量
num_qubits_per_qumode=3
alpha = np.sqrt(1.5)  # 光子数|α|²=2
chi_gf = 1.1  # CX门的χ匹配参数
num_cycles = 10  # 纠错周期数
cutoff=2**num_qubits_per_qumode
is_even=True
# --------------------------
# 初始化量子寄存器
# --------------------------
qumodes = c2.QumodeRegister(num_data, num_qubits_per_qumode, name="S")  # 数据qumode寄存器
ancillas = AncillaRegister(num_ancilla, name="A")  # 辅助qubit寄存器
temp_reg= ClassicalRegister(num_ancilla)
result_reg = ClassicalRegister(num_data*num_qubits_per_qumode, name="result")  # 结果经典寄存器
qc = c2.CVCircuit(qumodes, ancillas)

# 制备偶猫态
even_cat = prepare_cat_state(alpha, cutoff, is_even=True)
# 制备奇猫态
odd_cat = prepare_cat_state(alpha, cutoff, is_even=False)
# 绘制偶猫态光子数分布（仅偶数n有值）
plot_photon_distribution(even_cat, f"偶猫态（α=√1.5）光子数分布", cutoff)
# 绘制奇猫态光子数分布（仅奇数n有值）
plot_photon_distribution(odd_cat, f"奇猫态（α=√1.5）光子数分布", cutoff)

qc.add_register(temp_reg)
 # 对每个qumode初始化猫态
if is_even:
   for q in qumodes:
      qc.cv_initialize(even_cat, q)
else:
    for q in qumodes:
      qc.cv_initialize(odd_cat, q)

    #   qc.cv_measure(q,temp_reg)
    # _,_,counts=c2.util.simulate(qc,1024)
    # print(counts)


# --------------------------
# 完整纠错周期
# --------------------------
qc.add_register(result_reg)
for cycle in range(num_cycles):

    # 测量所有稳定器
    for i in range(num_ancilla):
        qc = measure_stabilizer(qc, data_idx=i, ancilla_idx=i,cutoff=cutoff,temp_reg=temp_reg)

# --------------------------
# 8. 最终测量
# --------------------------
for i in range(num_data):
    qc.cv_measure(qumodes[i], result_reg[i*num_qubits_per_qumode:i*num_qubits_per_qumode+num_qubits_per_qumode])

# --------------------------
# 9. 模拟执行与结果分析
# --------------------------

noise_model = create_two_photon_noise_model(cutoff, gamma=0.1)

state,result,_ = c2.util.simulate(
    cvcircuit=qc,
    shots=1024,
    noise_model=noise_model
)

counts = result.get_counts()

print("测量结果统计：")
print(counts)
total_shots = sum(counts.values())

