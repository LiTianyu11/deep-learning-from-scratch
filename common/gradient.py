# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


# def numerical_gradient(f, x):
#     h = 1e-4 # 0.0001
#     grad = np.zeros_like(x)
    
#     it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
#     while not it.finished:
#         idx = it.multi_index
#         tmp_val = x[idx]
#         x[idx] = tmp_val + h
#         fxh1 = f(x) # f(x+h)
        
#         x[idx] = tmp_val - h 
#         fxh2 = f(x) # f(x-h)
#         grad[idx] = (fxh1 - fxh2) / (2*h)
        
#         x[idx] = tmp_val # 値を元に戻す
#         it.iternext()   
        
#     return grad



def numerical_gradient(f, x):
    """
    数值微分（中心差分法）计算函数 f 在 x 处的梯度
    """
    h = 1e-4  # 微小增量 0.0001
    grad = np.zeros_like(x)

    # 遍历数组每一个元素的索引（现代写法，替代 np.nditer）
    # 2 * 3
    # (0, 0)
    # (0, 1)
    # (0, 2)
    # (1, 0)
    # (1, 1)
    # (1, 2)
    for idx in np.ndindex(x.shape):
        # 保存原始值
        original_val = x[idx]

        # f(x + h)
        x[idx] = original_val + h
        fx_plus = f(x)

        # f(x - h)
        x[idx] = original_val - h
        fx_minus = f(x)

        # 计算梯度
        grad[idx] = (fx_plus - fx_minus) / (2 * h)

        # 恢复原始值
        x[idx] = original_val

    return grad