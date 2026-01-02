"""
STATISTICAL ANALYSIS MODULE - 30 Concepts
Z-Score, Hurst Exponent, Correlation, Regression, Risk Metrics
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

# ==================== STATISTICAL INDICATORS (15) ====================

def calculate_z_score(prices: List[float], period: int = 20) -> Optional[float]:
    """Calculate Z-Score - Standard deviations from mean"""
    if len(prices) < period:
        return None
    mean = sum(prices[-period:]) / period
    std = np.std(prices[-period:])
    if std == 0:
        return 0
    return round((prices[-1] - mean) / std, 2)

def calculate_hurst_exponent(prices: List[float], max_lag: int = 20) -> Optional[float]:
    """Calculate Hurst Exponent - Trend persistence measure"""
    if len(prices) < max_lag * 2:
        return None
    try:
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return round(poly[0], 3)
    except:
        return 0.5

def calculate_autocorrelation(prices: List[float], lag: int = 1) -> Optional[float]:
    """Calculate Autocorrelation at given lag"""
    if len(prices) < lag + 10:
        return None
    mean = sum(prices) / len(prices)
    var = sum((p - mean) ** 2 for p in prices)
    if var == 0:
        return 0
    cov = sum((prices[i] - mean) * (prices[i - lag] - mean) for i in range(lag, len(prices)))
    return round(cov / var, 4)

def calculate_variance(prices: List[float], period: int = 20) -> Optional[float]:
    """Calculate Price Variance"""
    if len(prices) < period:
        return None
    return round(np.var(prices[-period:]), 8)

def calculate_skewness(prices: List[float], period: int = 20) -> Optional[float]:
    """Calculate Distribution Skewness"""
    if len(prices) < period:
        return None
    data = prices[-period:]
    mean = sum(data) / period
    std = np.std(data)
    if std == 0:
        return 0
    skew = sum((x - mean) ** 3 for x in data) / (period * std ** 3)
    return round(skew, 4)

def calculate_kurtosis(prices: List[float], period: int = 20) -> Optional[float]:
    """Calculate Distribution Kurtosis"""
    if len(prices) < period:
        return None
    data = prices[-period:]
    mean = sum(data) / period
    std = np.std(data)
    if std == 0:
        return 0
    kurt = sum((x - mean) ** 4 for x in data) / (period * std ** 4) - 3
    return round(kurt, 4)

def calculate_covariance(prices1: List[float], prices2: List[float], period: int = 20) -> Optional[float]:
    """Calculate Covariance between two price series"""
    if len(prices1) < period or len(prices2) < period:
        return None
    data1, data2 = prices1[-period:], prices2[-period:]
    mean1, mean2 = sum(data1) / period, sum(data2) / period
    cov = sum((data1[i] - mean1) * (data2[i] - mean2) for i in range(period)) / period
    return round(cov, 8)

def calculate_correlation(prices1: List[float], prices2: List[float], period: int = 20) -> Optional[float]:
    """Calculate Pearson Correlation Coefficient"""
    if len(prices1) < period or len(prices2) < period:
        return None
    data1, data2 = prices1[-period:], prices2[-period:]
    mean1, mean2 = sum(data1) / period, sum(data2) / period
    std1, std2 = np.std(data1), np.std(data2)
    if std1 == 0 or std2 == 0:
        return 0
    cov = sum((data1[i] - mean1) * (data2[i] - mean2) for i in range(period)) / period
    return round(cov / (std1 * std2), 4)

def calculate_rolling_correlation(prices1: List[float], prices2: List[float], period: int = 20) -> List[float]:
    """Calculate Rolling Correlation"""
    if len(prices1) < period or len(prices2) < period:
        return []
    correlations = []
    for i in range(period, min(len(prices1), len(prices2)) + 1):
        corr = calculate_correlation(prices1[:i], prices2[:i], period)
        correlations.append(corr if corr else 0)
    return correlations

def calculate_linear_regression(prices: List[float], period: int = 20) -> Dict:
    """Calculate Linear Regression Line"""
    if len(prices) < period:
        return {}
    y = prices[-period:]
    x = list(range(period))
    x_mean, y_mean = sum(x) / period, sum(y) / period
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(period))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(period))
    if denominator == 0:
        return {"slope": 0, "intercept": y_mean, "r_squared": 0}
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    y_pred = [slope * x[i] + intercept for i in range(period)]
    ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(period))
    ss_tot = sum((y[i] - y_mean) ** 2 for i in range(period))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return {"slope": round(slope, 6), "intercept": round(intercept, 5), "r_squared": round(r_squared, 4),
            "current_value": round(slope * (period - 1) + intercept, 5), "trend": "bullish" if slope > 0 else "bearish"}

def calculate_polynomial_regression(prices: List[float], period: int = 20, degree: int = 2) -> Dict:
    """Calculate Polynomial Regression"""
    if len(prices) < period:
        return {}
    y = prices[-period:]
    x = list(range(period))
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    y_pred = [poly(i) for i in x]
    y_mean = sum(y) / period
    ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(period))
    ss_tot = sum((y[i] - y_mean) ** 2 for i in range(period))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return {"coefficients": [round(c, 8) for c in coeffs], "r_squared": round(r_squared, 4),
            "current_value": round(poly(period - 1), 5), "next_value": round(poly(period), 5)}

def calculate_standard_error(prices: List[float], period: int = 20) -> Optional[float]:
    """Calculate Standard Error of the Mean"""
    if len(prices) < period:
        return None
    std = np.std(prices[-period:])
    return round(std / np.sqrt(period), 6)

def calculate_confidence_interval(prices: List[float], period: int = 20, confidence: float = 0.95) -> Dict:
    """Calculate Confidence Interval for Mean"""
    if len(prices) < period:
        return {}
    mean = sum(prices[-period:]) / period
    std_err = calculate_standard_error(prices, period)
    if std_err is None:
        return {}
    z_score = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
    margin = z_score * std_err
    return {"mean": round(mean, 5), "lower": round(mean - margin, 5), "upper": round(mean + margin, 5),
            "confidence": confidence}

def calculate_mean_reversion_probability(prices: List[float], period: int = 20) -> Optional[float]:
    """Calculate Mean Reversion Probability"""
    z = calculate_z_score(prices, period)
    if z is None:
        return None
    prob = min(100, abs(z) * 25)
    return round(prob, 1)

def calculate_trend_strength_statistical(prices: List[float], period: int = 20) -> Dict:
    """Calculate Statistical Trend Strength"""
    if len(prices) < period:
        return {}
    reg = calculate_linear_regression(prices, period)
    hurst = calculate_hurst_exponent(prices, min(period, 20))
    autocorr = calculate_autocorrelation(prices, 1)
    strength = 0
    if reg.get("r_squared", 0) > 0.7:
        strength += 30
    if hurst and hurst > 0.5:
        strength += 30
    if autocorr and autocorr > 0.5:
        strength += 20
    if reg.get("slope", 0) != 0:
        strength += 20
    return {"strength": strength, "r_squared": reg.get("r_squared"), "hurst": hurst, "autocorr": autocorr,
            "trend": reg.get("trend", "neutral")}

# ==================== RISK METRICS (15) ====================

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02, period: int = 252) -> Optional[float]:
    """Calculate Sharpe Ratio"""
    if len(returns) < 10:
        return None
    mean_return = sum(returns) / len(returns) * period
    std_return = np.std(returns) * np.sqrt(period)
    if std_return == 0:
        return 0
    return round((mean_return - risk_free_rate) / std_return, 2)

def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02, period: int = 252) -> Optional[float]:
    """Calculate Sortino Ratio - Downside risk adjusted"""
    if len(returns) < 10:
        return None
    mean_return = sum(returns) / len(returns) * period
    downside_returns = [r for r in returns if r < 0]
    if not downside_returns:
        return 10.0
    downside_std = np.std(downside_returns) * np.sqrt(period)
    if downside_std == 0:
        return 10.0
    return round((mean_return - risk_free_rate) / downside_std, 2)

def calculate_max_drawdown(prices: List[float]) -> Dict:
    """Calculate Maximum Drawdown"""
    if len(prices) < 2:
        return {}
    peak = prices[0]
    max_dd = 0
    max_dd_start, max_dd_end = 0, 0
    current_dd_start = 0
    for i, price in enumerate(prices):
        if price > peak:
            peak = price
            current_dd_start = i
        dd = (peak - price) / peak * 100
        if dd > max_dd:
            max_dd = dd
            max_dd_start = current_dd_start
            max_dd_end = i
    return {"max_drawdown": round(max_dd, 2), "start_idx": max_dd_start, "end_idx": max_dd_end,
            "recovery_needed": round(max_dd / (100 - max_dd) * 100, 2) if max_dd < 100 else 100}

def calculate_calmar_ratio(returns: List[float], prices: List[float], period: int = 252) -> Optional[float]:
    """Calculate Calmar Ratio - Return / Max Drawdown"""
    if len(returns) < 10 or len(prices) < 10:
        return None
    annual_return = sum(returns) / len(returns) * period * 100
    dd = calculate_max_drawdown(prices)
    max_dd = dd.get("max_drawdown", 1)
    if max_dd == 0:
        return 10.0
    return round(annual_return / max_dd, 2)

def calculate_beta(asset_returns: List[float], market_returns: List[float]) -> Optional[float]:
    """Calculate Beta - Market sensitivity"""
    if len(asset_returns) < 10 or len(market_returns) < 10:
        return None
    min_len = min(len(asset_returns), len(market_returns))
    asset_returns = asset_returns[-min_len:]
    market_returns = market_returns[-min_len:]
    cov = calculate_covariance(asset_returns, market_returns, min_len)
    var_market = calculate_variance(market_returns, min_len)
    if cov is None or var_market is None or var_market == 0:
        return 1.0
    return round(cov / var_market, 2)

def calculate_alpha(asset_returns: List[float], market_returns: List[float], risk_free_rate: float = 0.02) -> Optional[float]:
    """Calculate Alpha - Excess return"""
    if len(asset_returns) < 10 or len(market_returns) < 10:
        return None
    beta = calculate_beta(asset_returns, market_returns)
    if beta is None:
        return None
    asset_mean = sum(asset_returns) / len(asset_returns) * 252
    market_mean = sum(market_returns) / len(market_returns) * 252
    alpha = asset_mean - (risk_free_rate + beta * (market_mean - risk_free_rate))
    return round(alpha * 100, 2)

def calculate_information_ratio(asset_returns: List[float], benchmark_returns: List[float]) -> Optional[float]:
    """Calculate Information Ratio"""
    if len(asset_returns) < 10 or len(benchmark_returns) < 10:
        return None
    min_len = min(len(asset_returns), len(benchmark_returns))
    excess_returns = [asset_returns[i] - benchmark_returns[i] for i in range(min_len)]
    mean_excess = sum(excess_returns) / len(excess_returns)
    std_excess = np.std(excess_returns)
    if std_excess == 0:
        return 0
    return round(mean_excess / std_excess * np.sqrt(252), 2)

def calculate_treynor_ratio(returns: List[float], market_returns: List[float], risk_free_rate: float = 0.02) -> Optional[float]:
    """Calculate Treynor Ratio"""
    if len(returns) < 10:
        return None
    beta = calculate_beta(returns, market_returns)
    if beta is None or beta == 0:
        return None
    mean_return = sum(returns) / len(returns) * 252
    return round((mean_return - risk_free_rate) / beta, 2)

def calculate_var(returns: List[float], confidence: float = 0.95) -> Optional[float]:
    """Calculate Value at Risk (VaR)"""
    if len(returns) < 10:
        return None
    sorted_returns = sorted(returns)
    index = int((1 - confidence) * len(sorted_returns))
    return round(sorted_returns[index] * 100, 2)

def calculate_cvar(returns: List[float], confidence: float = 0.95) -> Optional[float]:
    """Calculate Conditional VaR (Expected Shortfall)"""
    if len(returns) < 10:
        return None
    sorted_returns = sorted(returns)
    index = int((1 - confidence) * len(sorted_returns))
    tail_returns = sorted_returns[:index + 1]
    if not tail_returns:
        return None
    return round(sum(tail_returns) / len(tail_returns) * 100, 2)

def calculate_omega_ratio(returns: List[float], threshold: float = 0) -> Optional[float]:
    """Calculate Omega Ratio"""
    if len(returns) < 10:
        return None
    gains = sum(r - threshold for r in returns if r > threshold)
    losses = sum(threshold - r for r in returns if r < threshold)
    if losses == 0:
        return 10.0
    return round(gains / losses, 2)

def calculate_gain_loss_ratio(returns: List[float]) -> Optional[float]:
    """Calculate Gain/Loss Ratio"""
    if len(returns) < 10:
        return None
    gains = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    if not gains or not losses:
        return None
    avg_gain = sum(gains) / len(gains)
    avg_loss = abs(sum(losses) / len(losses))
    if avg_loss == 0:
        return 10.0
    return round(avg_gain / avg_loss, 2)

def calculate_win_rate(returns: List[float]) -> Optional[float]:
    """Calculate Win Rate"""
    if len(returns) < 10:
        return None
    wins = sum(1 for r in returns if r > 0)
    return round(wins / len(returns) * 100, 1)

def calculate_profit_factor(returns: List[float]) -> Optional[float]:
    """Calculate Profit Factor"""
    if len(returns) < 10:
        return None
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    if gross_loss == 0:
        return 10.0
    return round(gross_profit / gross_loss, 2)

def calculate_ulcer_index(prices: List[float], period: int = 14) -> Optional[float]:
    """Calculate Ulcer Index - Downside volatility"""
    if len(prices) < period:
        return None
    max_price = prices[-period]
    squared_dd_sum = 0
    for i in range(-period, 0):
        if prices[i] > max_price:
            max_price = prices[i]
        dd = (prices[i] - max_price) / max_price * 100
        squared_dd_sum += dd ** 2
    return round(np.sqrt(squared_dd_sum / period), 2)

# ==================== MASTER ANALYSIS FUNCTION ====================

def analyze_all_statistics(prices: List[float], returns: List[float] = None, market_returns: List[float] = None) -> Dict:
    """Comprehensive Statistical Analysis - 30 concepts"""
    if returns is None:
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))] if len(prices) > 1 else []
    if market_returns is None:
        market_returns = returns
    
    results = {
        "distribution": {},
        "regression": {},
        "risk_metrics": {},
        "signals": {"bullish": 0, "bearish": 0},
        "overall_bias": "neutral",
        "strength": 50
    }
    
    # Distribution Statistics
    results["distribution"]["z_score"] = calculate_z_score(prices)
    results["distribution"]["hurst"] = calculate_hurst_exponent(prices)
    results["distribution"]["autocorrelation"] = calculate_autocorrelation(prices)
    results["distribution"]["variance"] = calculate_variance(prices)
    results["distribution"]["skewness"] = calculate_skewness(prices)
    results["distribution"]["kurtosis"] = calculate_kurtosis(prices)
    results["distribution"]["mean_reversion_prob"] = calculate_mean_reversion_probability(prices)
    results["distribution"]["confidence_interval"] = calculate_confidence_interval(prices)
    
    # Regression Analysis
    results["regression"]["linear"] = calculate_linear_regression(prices)
    results["regression"]["polynomial"] = calculate_polynomial_regression(prices)
    results["regression"]["trend_strength"] = calculate_trend_strength_statistical(prices)
    
    # Risk Metrics
    results["risk_metrics"]["sharpe"] = calculate_sharpe_ratio(returns)
    results["risk_metrics"]["sortino"] = calculate_sortino_ratio(returns)
    results["risk_metrics"]["max_drawdown"] = calculate_max_drawdown(prices)
    results["risk_metrics"]["calmar"] = calculate_calmar_ratio(returns, prices)
    results["risk_metrics"]["beta"] = calculate_beta(returns, market_returns)
    results["risk_metrics"]["alpha"] = calculate_alpha(returns, market_returns)
    results["risk_metrics"]["var_95"] = calculate_var(returns, 0.95)
    results["risk_metrics"]["cvar_95"] = calculate_cvar(returns, 0.95)
    results["risk_metrics"]["omega"] = calculate_omega_ratio(returns)
    results["risk_metrics"]["gain_loss"] = calculate_gain_loss_ratio(returns)
    results["risk_metrics"]["win_rate"] = calculate_win_rate(returns)
    results["risk_metrics"]["profit_factor"] = calculate_profit_factor(returns)
    results["risk_metrics"]["ulcer_index"] = calculate_ulcer_index(prices)
    
    # Calculate signals
    z = results["distribution"].get("z_score")
    if z:
        if z < -2:
            results["signals"]["bullish"] += 2
        elif z < -1:
            results["signals"]["bullish"] += 1
        elif z > 2:
            results["signals"]["bearish"] += 2
        elif z > 1:
            results["signals"]["bearish"] += 1
    
    hurst = results["distribution"].get("hurst")
    if hurst:
        if hurst > 0.5:
            trend = results["regression"].get("linear", {}).get("trend")
            if trend == "bullish":
                results["signals"]["bullish"] += 1
            elif trend == "bearish":
                results["signals"]["bearish"] += 1
    
    reg = results["regression"].get("linear", {})
    if reg.get("r_squared", 0) > 0.7:
        if reg.get("slope", 0) > 0:
            results["signals"]["bullish"] += 1
        else:
            results["signals"]["bearish"] += 1
    
    # Overall bias
    total = results["signals"]["bullish"] + results["signals"]["bearish"]
    if total > 0:
        if results["signals"]["bullish"] > results["signals"]["bearish"]:
            results["overall_bias"] = "BULLISH"
            results["strength"] = round(results["signals"]["bullish"] / total * 100, 1)
        elif results["signals"]["bearish"] > results["signals"]["bullish"]:
            results["overall_bias"] = "BEARISH"
            results["strength"] = round(results["signals"]["bearish"] / total * 100, 1)
    
    return results
