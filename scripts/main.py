import requests
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import configparser
import sys
import numpy as np
import matplotlib.patheffects as path_effects
import yfinance as yf
import urllib3
from datetime import datetime
import pytz

# <<< 新增: 动态构建路径 >>>
# 获取当前脚本所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取根目录 (scripts目录的上一级)
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
# 定义数据目录
DATA_DIR = os.path.join(ROOT_DIR, 'data')
# 确保data目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# 禁用在代理模式下可能出现的 InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================================================================
# 全局时区设置 - 美东时区 (ET)
# ==============================================================================
ET_TIMEZONE = pytz.timezone('America/New_York')


def get_et_now():
    """获取当前美东时间"""
    return datetime.now(ET_TIMEZONE)


def get_et_date_string():
    """获取当前美东日期字符串 (YYYY-MM-DD)"""
    return get_et_now().strftime('%Y-%m-%d')


def get_et_datetime_string():
    """获取当前美东日期时间字符串 (YYYY-MM-DD HH:MM:SS ET)"""
    return get_et_now().strftime('%Y-%m-%d %H:%M:%S %Z')


# ==============================================================================
# 1. 配置加载模块
# ==============================================================================

def load_config():
    """
    从 config.ini 文件加载所有配置。
    """
    config_file = os.path.join(ROOT_DIR, 'config.ini')
    if not os.path.exists(config_file):
        print(f"错误: 配置文件 '{config_file}' 不存在，请先创建。")
        sys.exit()

    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read(config_file, encoding='utf-8')

    try:
        # [General] & [Settings]
        data_source = config.getint('General', 'data_source', fallback=0)
        api_key = config.get('General', 'api_key', fallback=None)
        history_file = os.path.join(DATA_DIR, config.get('General', 'history_file'))
        plot_file = os.path.join(DATA_DIR, config.get('General', 'plot_file'))
        pie_chart_file = os.path.join(DATA_DIR, config.get('General', 'pie_chart_file'))
        max_retries = config.getint('Settings', 'max_retries')
        retry_delay = config.getint('Settings', 'retry_delay_seconds')

        if data_source == 1 and (not api_key or api_key == 'YOUR_API_KEY_HERE'):
            print("错误: data_source 设置为 1 (Alpha Vantage)，但未提供有效的 api_key。")
            sys.exit()

        # [Portfolio] - 股票
        portfolio = []
        for ticker, quantity in config.items('Portfolio'):
            portfolio.append((ticker.upper(), float(quantity)))

        # [OptionsPortfolio] - 期权
        options_portfolio = []
        if config.has_section('OptionsPortfolio'):
            for key, quantity_str in config.items('OptionsPortfolio'):
                try:
                    parts = key.upper().rsplit('_', 2)
                    if len(parts) != 3:
                        continue
                    base, strike_str, opt_type = parts
                    date_parts = base.rsplit('_', 1)
                    if len(date_parts) != 2:
                        continue
                    ticker, date_str = date_parts
                    if opt_type not in ['CALL', 'PUT']:
                        continue
                    option_details = {
                        'key': key.upper(),
                        'ticker': ticker,
                        'expiry': date_str,
                        'strike': float(strike_str),
                        'type': opt_type,
                        'quantity': float(quantity_str)
                    }
                    options_portfolio.append(option_details)
                except Exception:
                    print(f"警告: 无法解析期权 '{key}'。")

        # [Cash]
        cash_amount = config.getfloat('Cash', 'amount', fallback=0.0)

        return (data_source, api_key, history_file, plot_file, pie_chart_file,
                max_retries, retry_delay, portfolio, options_portfolio, cash_amount)

    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
        print(f"错误: 配置文件 'config.ini' 格式不正确或缺少必要项: {e}")
        sys.exit()


# 在程序开始时加载所有配置
(DATA_SOURCE, API_KEY, HISTORY_FILE, PLOT_FILE, PIE_CHART_FILE, MAX_RETRIES,
 RETRY_DELAY, portfolio, options_portfolio, CASH_AMOUNT) = load_config()


# ==============================================================================
# 2. 获取资产价格的核心函数 (重构版 - 支持盘前盘后 + 美东时区)
# ==============================================================================

def get_stock_price_alphavantage(ticker):
    """
    使用 Alpha Vantage API 获取股票价格
    """
    av_ticker = ticker.replace('-', '.')
    print(f"  - [AlphaVantage] 正在获取 {av_ticker}...")
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={av_ticker}&apikey={API_KEY}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        global_quote = data.get('Global Quote')
        if global_quote and '05. price' in global_quote and '07. latest trading day' in global_quote:
            time.sleep(1)
            price = float(global_quote['05. price'])
            trading_day = global_quote['07. latest trading day']
            return price, trading_day
        else:
            print(f"  - 警告: AlphaVantage 未能找到 '{av_ticker}' 的数据。 响应: {data}")
            time.sleep(1)
            return None
    except Exception as e:
        print(f"  - 错误: 请求 '{av_ticker}' 时发生网络错误: {e}")
        return None


def get_stock_price_yfinance(ticker):
    """
    获取股票价格的改进版本，智能判断市场状态。
    支持盘前、盘中、盘后价格，所有时间基于美东时区。
    """
    yf_ticker = ticker.replace('.', '-')
    price_type = "未知"
    price = None
    trading_day = None

    # ===== 第一步：尝试从 info 获取实时价格 =====
    try:
        print(f"  - [yfinance-info] 正在获取 {yf_ticker} 的实时报价...")
        stock = yf.Ticker(yf_ticker)
        info = stock.info

        # 获取市场状态
        market_state = info.get('marketState', 'CLOSED')  # 可能值: PRE, REGULAR, POST, CLOSED
        print(f"    -> 市场状态: {market_state}")

        # 根据市场状态智能选择价格
        if market_state == 'PRE':
            # 盘前时段：优先使用盘前价格
            price = info.get('preMarketPrice')
            price_type = "盘前价"
            # 使用盘前时间对应的交易日
            pre_market_time = info.get('preMarketTime')
            if pre_market_time:
                trading_day = datetime.fromtimestamp(pre_market_time, ET_TIMEZONE).strftime('%Y-%m-%d')

        elif market_state == 'REGULAR':
            # 盘中时段：使用常规市价
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            price_type = "盘中价"
            regular_market_time = info.get('regularMarketTime')
            if regular_market_time:
                trading_day = datetime.fromtimestamp(regular_market_time, ET_TIMEZONE).strftime('%Y-%m-%d')

        elif market_state == 'POST':
            # 盘后时段：优先使用盘后价格
            price = info.get('regularMarketPrice') or info.get('currentPrice')  # 改为读取 regularMarketPrice
            price_type = "收盘价"  # 修改标签为"收盘价"
            regular_market_time = info.get('regularMarketTime')  # 使用 regularMarketTime 而不是 postMarketTime
            if regular_market_time:
                trading_day = datetime.fromtimestamp(regular_market_time, ET_TIMEZONE).strftime('%Y-%m-%d')

        # 兜底逻辑：如果上述都没获取到价格
        if price is None:
            price = (info.get('regularMarketPrice') or
                     info.get('previousClose') or
                     info.get('currentPrice'))
            price_type = "最近价格"
            # 使用 regularMarketTime 或当前美东日期
            regular_market_time = info.get('regularMarketTime')
            if regular_market_time:
                trading_day = datetime.fromtimestamp(regular_market_time, ET_TIMEZONE).strftime('%Y-%m-%d')
            else:
                trading_day = get_et_date_string()

        if price is not None and trading_day is None:
            # 最后的保险：如果有价格但没日期
            trading_day = get_et_date_string()

        if price is not None:
            print(f"  - [yfinance-info] 成功获取 {yf_ticker} ({price_type}, 市场状态: {market_state})")
            return price, trading_day

    except Exception as e:
        print(f"  - [yfinance-info] 获取实时报价失败: {e}。将尝试备用方案。")

    # ===== 第二步：备用方案 - 使用 history 获取最近收盘价 =====
    print(f"  - [yfinance-history] 正在获取 {yf_ticker} 的最近收盘价...")
    try:
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(period='5d', auto_adjust=True)

        if hist.empty:
            raise ConnectionError(f"备用方案也返回了空数据 (可能是无效代码: {yf_ticker})")

        last_trade = hist.iloc[-1]
        price = float(last_trade['Close'])

        # 将时间戳转换为美东时区日期
        if last_trade.name.tzinfo is None:
            # 如果没有时区信息，假设是UTC
            trading_day = last_trade.name.tz_localize('UTC').tz_convert(ET_TIMEZONE).strftime('%Y-%m-%d')
        else:
            trading_day = last_trade.name.tz_convert(ET_TIMEZONE).strftime('%Y-%m-%d')

        price_type = "最近收盘价"
        print(f"  - [yfinance-history] 成功获取 {yf_ticker} ({price_type})")
        return price, trading_day

    except Exception as e_fallback:
        print(f"  - [yfinance-history] 失败: {e_fallback}")
        return None


def get_option_price_yfinance(ticker, expiry, strike_price, option_type):
    """
    使用 yfinance 获取期权价格
    所有时间基于美东时区
    """
    option_name = f"{ticker} {expiry} {strike_price} {option_type}"

    try:
        print(f"  - [yfinance-option] 正在获取期权 {option_name}...")
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiry)
        df = chain.calls if option_type == 'CALL' else chain.puts

        if df.empty:
            raise ValueError(f"未找到 {expiry} 的 {option_type} 期权链")

        contract = df[df['strike'] == strike_price]
        if contract.empty:
            raise ValueError(f"未找到行权价为 {strike_price} 的合约")

        price = float(contract.iloc[0]['lastPrice'])
        # 期权价格使用当前美东日期
        trading_day = get_et_date_string()

        print(f"  - [yfinance-option] 成功获取期权 {option_name}")
        return price, trading_day

    except Exception as e:
        print(f"  - [yfinance-option] 失败: {e}")
        return None


# ==============================================================================
# 3. 获取历史价格函数 (基于美东时区)
# ==============================================================================

def get_historical_stock_price(ticker, target_date):
    """
    获取给定资产在特定日期的收盘价。
    智能处理股票和期权两种代码。
    所有日期基于美东时区。
    """
    api_ticker = ticker.replace('.', '-')

    # --- 智能识别并转换期权代码 ---
    if '_' in ticker:
        print(f"    -> 检测到期权代码 '{ticker}'，正在转换为yfinance格式...")
        try:
            parts = ticker.split('_')
            underlying = parts[0]
            date_parts = parts[1].split('-')
            strike = parts[2]
            option_type = parts[3][0].upper()  # 'PUT' -> 'P', 'CALL' -> 'C'

            # 格式化日期: 2026-01-16 -> 260116
            yy = date_parts[0][2:]
            mm = date_parts[1]
            dd = date_parts[2]

            # 格式化执行价: 130 -> 00130000 (8位，补零，包含3位小数)
            strike_price_formatted = f"{float(strike):.3f}".replace('.', '')
            strike_price_padded = strike_price_formatted.zfill(8)

            # 拼接成yfinance官方格式
            api_ticker = f"{underlying}{yy}{mm}{dd}{option_type}{strike_price_padded}"
            print(f"    -> 转换成功: '{api_ticker}'")

        except Exception as e:
            print(f"    -> 错误: 转换期权代码 '{ticker}' 失败: {e}")
            return None

    # --- 获取历史数据 ---
    try:
        stock = yf.Ticker(api_ticker)

        # 目标日期加一天，以确保能覆盖到当天的交易数据
        end_date = (pd.to_datetime(target_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        # 获取从目标日期开始的历史数据
        hist = stock.history(start=target_date, end=end_date, auto_adjust=False)

        if not hist.empty:
            # 优先使用 'Close'，如果不存在，则使用 'close'
            if 'Close' in hist.columns:
                return float(hist['Close'].iloc[0])
            elif 'close' in hist.columns:
                return float(hist['close'].iloc[0])
            else:
                print(f"    -> 警告: 在 {target_date} 的数据中找不到 'Close' 或 'close' 列。")
                return None
        else:
            print(f"    -> 警告: yfinance未能返回 {api_ticker} 在 {target_date} 的任何数据。")

            # 尝试获取期权的info，对于某些情况可能有效
            if '_' in ticker:
                info = stock.info
                if 'lastPrice' in info and info['lastPrice'] is not None:
                    print("    -> 备用方案: 从info中成功获取 'lastPrice'。")
                    return float(info['lastPrice'])

            return None

    except Exception as e:
        print(f"    -> 错误: 获取历史价格失败: {e}")
        print(f"    -> {api_ticker} 可能已退市或数据不可用")
        return None


# ==============================================================================
# 4. 计算总价值并收集价格 (基于美东时区)
# ==============================================================================

def calculate_portfolio_value():
    """
    计算总价值，并同时收集每个资产的 (总价值, 单价) 元组。
    所有日期基于美东时区。
    """
    total_value = 0.0
    asset_details = {}  # 将存储 (总价值, 单价) 的元组
    portfolio_date = None
    source_name = "yfinance" if DATA_SOURCE == 0 else "Alpha Vantage"

    print(f"\n{'=' * 70}")
    print(f"开始计算投资组合价值")
    print(f"当前美东时间: {get_et_datetime_string()}")
    print(f"{'=' * 70}\n")
    print(f"正在使用 [{source_name}] 获取您的股票价值...\n")

    # ===== 处理股票 =====
    for ticker, quantity in portfolio:
        price, fetched_date = 0.0, None
        result = None

        for attempt in range(MAX_RETRIES):
            get_price_func = get_stock_price_yfinance if DATA_SOURCE == 0 else get_stock_price_alphavantage
            result = get_price_func(ticker)

            if result:
                price, fetched_date = result
                if portfolio_date is None:
                    portfolio_date = fetched_date
                break

            if attempt < MAX_RETRIES - 1:
                print(f"  - 获取 {ticker} 失败。将在 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)

        stock_value = price * quantity
        asset_details[ticker] = (stock_value, price)
        total_value += stock_value

        if result:
            print(f"  -> ✓ 成功: {ticker} {quantity:g} 股 @ ${price:.2f} = ${stock_value:,.2f}")
        else:
            raise RuntimeError(f"经过 {MAX_RETRIES} 次尝试后，仍无法获取 {ticker} 的价格。")

    # ===== 处理期权 =====
    if DATA_SOURCE == 0 and options_portfolio:
        print("\n正在使用 [yfinance] 获取您的期权价值...\n")

        for opt in options_portfolio:
            price, fetched_date = 0.0, None
            result = None

            for attempt in range(MAX_RETRIES):
                result = get_option_price_yfinance(opt['ticker'], opt['expiry'], opt['strike'], opt['type'])

                if result:
                    price, fetched_date = result
                    if portfolio_date is None:
                        portfolio_date = fetched_date
                    break

                if attempt < MAX_RETRIES - 1:
                    print(f"  - 获取 {opt['key']} 失败。将在 {RETRY_DELAY} 秒后重试...")
                    time.sleep(RETRY_DELAY)

            option_value = price * opt['quantity'] * 100
            asset_details[opt['key']] = (option_value, price)
            total_value += option_value

            if result:
                print(f"  -> ✓ 成功: {opt['key']} {opt['quantity']:g} 张 @ ${price:.2f} = ${option_value:,.2f}")
            else:
                raise RuntimeError(f"经过 {MAX_RETRIES} 次尝试后，仍无法获取 {opt['key']} 的价格。")

    # ===== 处理现金 =====
    asset_details['CASH'] = (CASH_AMOUNT, 1.0)
    total_value += CASH_AMOUNT

    if CASH_AMOUNT > 0:
        print(f"\n计入现金余额: ${CASH_AMOUNT:,.2f}")

    # 如果没有获取到任何日期，使用当前美东日期
    if portfolio_date is None:
        portfolio_date = get_et_date_string()
        print(f"\n提示: 未能从API获取交易日期，使用当前美东日期: {portfolio_date}")

    print("\n" + "=" * 70)
    print(f"投资组合总价值 (截至 {portfolio_date}): ${total_value:,.2f}")
    print("=" * 70)

    return total_value, asset_details, portfolio_date


# ==============================================================================
# 5. 保存历史数据 (基于美东时区)
# ==============================================================================

def save_history(date_to_save, value_to_save, asset_details):
    """
    将当日的投资组合详情追加或更新到历史记录CSV文件中。
    每个资产单元格存储格式: '(总价值|单价)'
    所有日期基于美东时区。
    """
    # 1. 准备当日的新数据行
    new_row_data = {'total_value': f"{value_to_save:.2f}"}
    for asset, (val, price) in asset_details.items():
        new_row_data[asset] = f"({val:.2f}|{price:.2f})"

    new_row = pd.Series(new_row_data, name=date_to_save)

    # 2. 读取现有数据
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        try:
            df = pd.read_csv(HISTORY_FILE, index_col='date', dtype=str)
        except (pd.errors.EmptyDataError, IndexError):
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # 3. 覆盖当天数据
    if date_to_save in df.index:
        df = df.drop(index=date_to_save)
        print(f"\n提示: 日期 {date_to_save} 的旧数据已找到，将进行覆盖更新。")
    else:
        print(f"\n成功: 已将 {date_to_save} 的新数据添加到历史记录。")

    # 4. 添加新行并对齐
    df = pd.concat([df, new_row.to_frame().T])

    # 5. 用代表0的元组字符串填充新出现的NaN值
    df.fillna("(0.00|0.00)", inplace=True)

    # total_value列的NaN用 '0.00' 填充
    if 'total_value' in df.columns:
        df['total_value'].replace("(0.00|0.00)", "0.00", inplace=True)

    # 6. 重新排序列
    if 'total_value' in df.columns:
        cols = [c for c in df.columns if c != 'total_value']
        cols.sort()
        final_cols = ['total_value'] + cols
        df = df[final_cols]

    # 7. 保存（按日期降序排列）
    df.sort_index(ascending=False, inplace=True)
    df.to_csv(HISTORY_FILE, index=True, index_label='date')
    print(f"历史记录已成功更新到: {HISTORY_FILE}")


# ==============================================================================
# 6. 数据解析辅助函数
# ==============================================================================

def parse_value_from_cell(cell):
    """
    手动解析单元格的函数。
    能处理 '(价值|价格)' 格式和纯数字格式。
    """
    try:
        if isinstance(cell, str) and cell.strip().startswith('(') and cell.strip().endswith(')'):
            parts = cell.strip()[1:-1].split('|')
            if len(parts) == 2:
                return float(parts[0].strip())
            else:
                return 0.0
        return float(cell)
    except (ValueError, TypeError, AttributeError):
        return 0.0


# ==============================================================================
# 7. 绘制历史价值图表
# ==============================================================================

def plot_history_graph(output_filename):
    """
    绘制投资组合历史价值堆叠图
    """
    if not os.path.exists(HISTORY_FILE):
        print("找不到历史数据文件，无法绘制图表。")
        return

    print(f"\n正在生成历史趋势图...")

    # 读取原始数据
    df_raw = pd.read_csv(HISTORY_FILE, index_col='date', parse_dates=True)
    df_raw.sort_index(inplace=True)

    # 创建用于绘图的数值DataFrame
    df_plot = df_raw.map(parse_value_from_cell)

    if len(df_plot) < 1:
        print("历史数据不足，无法生成图表。")
        return

    asset_columns = [col for col in df_plot.columns if col != 'total_value']
    positive_assets = df_plot[asset_columns].clip(lower=0)

    # 计算累积值，用于定位标签
    df_cum = positive_assets.cumsum(axis=1)

    fig, ax = plt.subplots(figsize=(16, 9))

    # 绘制堆叠图
    colors = plt.cm.viridis(np.linspace(0, 1, len(asset_columns)))
    ax.stackplot(df_plot.index, positive_assets.T, labels=asset_columns, colors=colors)

    # 绘制总价值曲线
    ax.plot(df_plot.index, df_plot['total_value'], color='black', linewidth=2.5,
            linestyle='--', label='Total Value')

    # 在每个色块的起始位置添加标签
    text_effect = [path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()]
    for col in asset_columns:
        col_data = positive_assets[col]
        first_occurrence_index = col_data.ne(0).idxmax()

        if pd.isna(first_occurrence_index) or col_data.loc[first_occurrence_index] == 0:
            continue

        y_base = df_cum.loc[first_occurrence_index, col] - col_data.loc[first_occurrence_index]
        y_pos = y_base + 0.5 * col_data.loc[first_occurrence_index]

        ax.text(first_occurrence_index, y_pos, col, color='white', ha='left', va='center',
                fontsize=10, path_effects=text_effect, fontweight='bold')

    ax.set_title('Portfolio Value Over Time (ET)', fontsize=20, pad=20)
    ax.set_ylabel('Value ($)', fontsize=14)
    ax.set_xlabel('Date (ET)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%1.0f'))

    if not df_plot.index.empty:
        ax.set_xlim(df_plot.index[0], df_plot.index[-1])

    ax.axhline(0, color='black', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()

    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"✓ 成功: 历史趋势图已保存到 '{output_filename}'")
    except Exception as e:
        print(f"✗ 错误: 保存历史趋势图时出错: {e}")
    finally:
        plt.close(fig)


# ==============================================================================
# 8. 绘制当日仓位饼图
# ==============================================================================

def plot_pie_chart(asset_details, output_filename):
    """
    绘制当日资产配置饼图
    """
    print(f"\n正在生成当日仓位饼图...")

    # 提取价值 > 0 的资产
    asset_values = {k: v[0] for k, v in asset_details.items() if v[0] > 0}

    if not asset_values:
        print("没有正价值的持仓可用于生成饼图。")
        return

    labels = list(asset_values.keys())
    sizes = list(asset_values.values())
    total_size = sum(sizes)

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(aspect="equal"))

    # 绘制饼图
    wedges, texts = ax.pie(sizes,
                           startangle=90,
                           colors=plt.cm.viridis(np.linspace(0, 1, len(labels))),
                           radius=1.2)

    ax.set_title(f'Asset Composition (ET: {get_et_date_string()})', fontsize=20, pad=20)

    # 自定义内部标签
    text_effect = [path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()]
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta1 + wedge.theta2) / 2.
        x = wedge.r * 0.7 * np.cos(np.deg2rad(angle))
        y = wedge.r * 0.7 * np.sin(np.deg2rad(angle))

        percent = sizes[i] / total_size * 100
        label_text = f"{labels[i]}\n{percent:.1f}%"

        if percent < 4:
            label_text = f"{percent:.1f}%"

        ax.text(x, y, label_text,
                ha='center', va='center',
                color='white', fontsize=10,
                path_effects=text_effect, fontweight='bold')

    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"✓ 成功: 当日仓位饼图已保存到 '{output_filename}'")
    except Exception as e:
        print(f"✗ 错误: 保存仓位饼图时出错: {e}")
    finally:
        plt.close(fig)


# ==============================================================================
# 9. 历史数据校验与修复模块 (基于美东时区)
# ==============================================================================

def validate_and_repair_history():
    """
    校验并修复历史数据，并清理已售罄的资产列。
    所有日期操作基于美东时区。
    """
    if not os.path.exists(HISTORY_FILE):
        return

    print("\n" + "=" * 70)
    print("开始执行历史数据完整性校验...")
    print("=" * 70)

    try:
        df = pd.read_csv(HISTORY_FILE, index_col='date', dtype=str)
    except Exception as e:
        print(f"错误: 读取历史文件 '{HISTORY_FILE}' 失败: {e}")
        return

    df_repaired = df.copy()
    changes_made = False

    # 数据修复循环
    for date, row in df.iterrows():
        for ticker, cell_value in row.items():
            if ticker == 'total_value' or pd.isna(cell_value):
                continue

            # -------------------- 1. 现金 (CASH) 处理逻辑 --------------------
            if ticker == 'CASH':
                try:
                    is_tuple_format = isinstance(cell_value, str) and cell_value.strip().startswith('(')
                    cash_amount = 0.0
                    needs_fixing = False

                    if is_tuple_format:
                        parts = cell_value.strip()[1:-1].split('|')
                        cash_amount = float(parts[0].strip())
                        price = float(parts[1].strip()) if len(parts) > 1 else 0.0
                        if price != 1.0:
                            needs_fixing = True
                    else:
                        cash_amount = float(cell_value)
                        needs_fixing = True

                    if needs_fixing:
                        new_cash_value = f"({cash_amount:.2f}|1.00)"
                        if df_repaired.at[date, ticker] != new_cash_value:
                            df_repaired.at[date, ticker] = new_cash_value
                            changes_made = True
                            print(f"  - 修正 CASH 格式: {date} 从 '{cell_value}' -> '{new_cash_value}'")

                except (ValueError, IndexError):
                    print(f"  - 警告: 无法解析 CASH 单元格 {date}: '{cell_value}'")
                continue

            # -------------------- 2. 期权 (Options) 处理逻辑 --------------------
            elif '_' in ticker:
                total_val, price = 0.0, 0.0
                try:
                    if isinstance(cell_value, str) and cell_value.strip().startswith('('):
                        parts = cell_value.strip()[1:-1].split('|')
                        total_val = float(parts[0].strip())
                        price = float(parts[1].strip()) if len(parts) > 1 else 0.0
                    else:
                        total_val = float(cell_value)
                except (ValueError, IndexError):
                    print(f"  - 警告: 无法解析期权单元格 {ticker} {date}: '{cell_value}'")
                    continue

                if total_val == 0:
                    correct_format = "(0.00|0.00)"
                    if cell_value.strip() not in ["(0.00|0.00)", "(0.0|0.0)"]:
                        df_repaired.at[date, ticker] = correct_format
                        changes_made = True
                        print(f"  - 修正零值期权格式: {ticker} {date} 从 '{cell_value}' -> '{correct_format}'")

                elif total_val > 0 and price <= 0:
                    print(f"  - 发现不一致期权数据: {ticker} {date} [价值: {total_val:.2f}, 价格缺失]")
                    print(f"    -> 正在尝试获取 {date} 的历史价格...")
                    historical_price = get_historical_stock_price(ticker, date)
                    time.sleep(1)

                    if historical_price is not None:
                        new_cell_value = f"({total_val:.2f}|{historical_price:.2f})"
                        df_repaired.at[date, ticker] = new_cell_value
                        changes_made = True
                        print(f"    -> ✓ 成功修复期权: 价格更新为 ${historical_price:.2f}")
                    else:
                        print(f"    -> ✗ 修复失败: 未能获取到期权 {ticker} 在 {date} 的历史价格")
                continue

            # -------------------- 3. 股票 (Stock) 处理逻辑 --------------------
            else:
                total_val, price = 0.0, 0.0
                try:
                    if isinstance(cell_value, str) and cell_value.strip().startswith('('):
                        parts = cell_value.strip()[1:-1].split('|')
                        total_val = float(parts[0].strip())
                        price = float(parts[1].strip()) if len(parts) > 1 else 0.0
                    else:
                        total_val = float(cell_value)
                except (ValueError, IndexError):
                    continue

                if total_val > 0 and price <= 0:
                    print(f"  - 发现不一致股票数据: {ticker} {date} [价值: {total_val:.2f}, 价格缺失]")
                    print(f"    -> 正在尝试获取 {date} 的历史价格...")
                    historical_price = get_historical_stock_price(ticker, date)
                    time.sleep(1)

                    if historical_price is not None:
                        new_cell_value = f"({total_val:.2f}|{historical_price:.2f})"
                        df_repaired.at[date, ticker] = new_cell_value
                        changes_made = True
                        print(f"    -> ✓ 成功修复股票: 价格更新为 ${historical_price:.2f}")
                    else:
                        print(f"    -> ✗ 修复失败: 未能获取到股票 {ticker} 在 {date} 的历史价格")

    # 清理已清仓资产列
    columns_to_drop = []
    asset_columns = [col for col in df_repaired.columns if col != 'total_value']

    for col in asset_columns:
        try:
            total_asset_value = df_repaired[col].apply(parse_value_from_cell).sum()
            if total_asset_value == 0:
                columns_to_drop.append(col)
        except Exception:
            continue

    if columns_to_drop:
        df_repaired.drop(columns=columns_to_drop, inplace=True)
        changes_made = True
        print(f"\n信息: 检测到并清除了已售罄的资产列: {', '.join(columns_to_drop)}")

    # 保存修复后的数据
    if changes_made:
        print("\n校验完成。发现并修复/清理了数据，正在保存更新后的历史文件...")
        try:
            df_repaired.to_csv(HISTORY_FILE, index=True, index_label='date')
            print(f"✓ 成功: 已将更新后的历史数据保存到 '{HISTORY_FILE}'")
        except Exception as e:
            print(f"✗ 错误: 保存更新后的历史文件失败: {e}")
    else:
        print("\n校验完成。未发现需要修复或清理的数据。")

    print("=" * 70)


# ==============================================================================
# 10. 主执行逻辑
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("投资组合追踪系统 (Portfolio Tracker)")
    print("所有时间基于美东时区 (America/New_York)")
    print("=" * 70)

    # 计算投资组合价值
    final_total_value, all_asset_details, data_date = calculate_portfolio_value()

    if data_date is not None:
        # 保存历史数据
        save_history(data_date, final_total_value, all_asset_details)

        # 校验和修复历史数据
        validate_and_repair_history()

        # 生成图表
        plot_history_graph(PLOT_FILE)
        plot_pie_chart(all_asset_details, PIE_CHART_FILE)

        print("\n" + "=" * 70)
        print(f"✓ 所有任务完成! (美东时间: {get_et_datetime_string()})")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ 错误: 未能获取到任何有效的交易日期，无法保存和绘图。")
        print("=" * 70)
