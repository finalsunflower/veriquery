"""
Common Chip Pinout Library.

Provides built-in pinout configurations for widely-used chips as a fallback
when the knowledge graph has no data.  The primary consumer is
``agents/workflow_nodes.py → PinoutNode``.

Lookup strategy inside ``get_pinout()``:
  1. Normalize the chip name (strip manufacturer prefix / package suffix).
  2. Exact lookup in ``STANDARD_PINOUTS``.
  3. Alias lookup in ``PINOUT_ALIASES`` → re-query ``STANDARD_PINOUTS``.
  4. Prefix-based fuzzy match on both dicts.
  5. Return ``None`` if all levels miss.
"""

import re
from typing import Dict, Optional


class CommonPinoutLibrary:
    """Built-in pinout configurations for common chips.

    All data lives in class-level dicts (``PINOUT_ALIASES``,
    ``STANDARD_PINOUTS``) and every method is a ``@classmethod``,
    so no instantiation is needed.
    """

    _SUFFIX_PATTERN = re.compile(r'[A-Z]$')

    PINOUT_ALIASES: Dict[str, str] = {
        #   - UA = Fairchild（仙童）
        #   - SE/RC = Signetics/Philips
        #   - TLC = TI 的 LinCMOS 系列
        #   - LM = National Semiconductor（国家半导体）
        #   - NE = Signetics/Philips
        #   - MC = Motorola（摩托罗拉）
        #   - L = STMicroelectronics（意法半导体）
        #   - AMS = Advanced Monolithic Systems
        #
        #      - "BluePill" → "STM32F103C8T6"（开发板昵称，非厂商前缀）
        #      - "ArduinoUno" → "ATmega328P"（平台名到芯片的映射）
        #      - "ULN2004" → "ULN2003"（不同型号但引脚兼容）
        #      - "AMS1117" → "AMS1117-3.3"（默认指向最常用版本）
        #      因此需要显式的别名映射表来覆盖这些特殊情况
        #
        #      查询时：先查 STANDARD_PINOUTS，未命中再查 PINOUT_ALIASES 获取标准名后重查

        # ----- 74HC04 六反相器 -----
        # 标准名: 74HC04 | 别名: HC04(省略74前缀), SN74HC04(TI), CD74HC04(Harris),
        #                   74HCT04(TTL电平兼容版), HCT04, SN74HCT04, CD74HCT04
        "HC04": "74HC04",
        "SN74HC04": "74HC04",
        "CD74HC04": "74HC04",
        "74HCT04": "74HC04",
        "HCT04": "74HC04",
        "SN74HCT04": "74HC04",
        "CD74HCT04": "74HC04",

        # ----- 74HC595 8位移位寄存器 -----
        # 标准名: 74HC595 | 别名: HC595(省略74), SN74HC595(TI), CD74HC595(Harris)
        "HC595": "74HC595",
        "SN74HC595": "74HC595",
        "CD74HC595": "74HC595",

        # ----- 74LS00 四2输入与非门 -----
        # 标准名: 74LS00 | 别名: LS00(省略74), SN74LS00(TI), HD74LS00(Hitachi),
        #                   HC00/74HC00(CMOS版，引脚兼容)
        "LS00": "74LS00",
        "SN74LS00": "74LS00",
        "HD74LS00": "74LS00",
        "HC00": "74LS00",
        "74HC00": "74LS00",

        # ----- NE555 定时器 -----
        # 标准名: NE555 | 别名: 555(省略NE), LM555(NSC), TLC555(TI LinCMOS版)
        #           TLC555是CMOS版本，功耗更低但功能引脚完全相同
        "555": "NE555",
        "LM555": "NE555",
        "TLC555": "NE555",

        # ----- LM358 双运算放大器 -----
        # 标准名: LM358 | 别名: 358(省略LM), UA358(Fairchild)
        "358": "LM358",
        "UA358": "LM358",

        # ----- NE5532 双低噪声运算放大器 -----
        # 标准名: NE5532 | 别名: 5532(省略NE), SE5532(Signetics), RC5532(Raytheon), LM5532(NSC)
        "5532": "NE5532",
        "SE5532": "NE5532",
        "RC5532": "NE5532",
        "LM5532": "NE5532",

        # ----- 74HC165 8位并入串出移位寄存器 -----
        # 标准名: 74HC165 | 别名: HC165(省略74), SN74HC165(TI), CD74HC165(Harris)
        "HC165": "74HC165",
        "SN74HC165": "74HC165",
        "CD74HC165": "74HC165",

        # ----- 74HC245 八路总线收发器 -----
        # 标准名: 74HC245 | 别名: HC245(省略74), SN74HC245(TI), CD74HC245(Harris),
        #                     74LVC245/LVC245(3.3V低电压版，引脚兼容)
        #           LVC系列是3.3V版本，引脚排列与HC完全相同
        "HC245": "74HC245",
        "SN74HC245": "74HC245",
        "CD74HC245": "74HC245",
        "74LVC245": "74HC245",
        "LVC245": "74HC245",

        # ----- 74HC138 3-8线译码器 -----
        # 标准名: 74HC138 | 别名: HC138(省略74), SN74HC138(TI), CD74HC138(Harris)
        #           常用于地址译码、片选信号生成
        "HC138": "74HC138",
        "SN74HC138": "74HC138",
        "CD74HC138": "74HC138",

        # ----- 74HC126 四路三态缓冲器 -----
        # 标准名: 74HC126 | 别名: HC126(省略74), SN74HC126(TI)
        #           三态输出可实现总线共享，OE=0时输出高阻态
        "HC126": "74HC126",
        "SN74HC126": "74HC126",

        # ----- 74HC08 四2输入与门 -----
        # 标准名: 74HC08 | 别名: HC08(省略74), SN74HC08(TI)
        "HC08": "74HC08",
        "SN74HC08": "74HC08",

        # ----- 74HC02 四2输入或非门 -----
        # 标准名: 74HC02 | 别名: HC02(省略74), SN74HC02(TI)
        "HC02": "74HC02",
        "SN74HC02": "74HC02",

        # ----- 74HC32 四2输入或门 -----
        # 标准名: 74HC32 | 别名: HC32(省略74), SN74HC32(TI)
        "HC32": "74HC32",
        "SN74HC32": "74HC32",

        # ----- 74HC164 8位串入并出移位寄存器 -----
        # 标准名: 74HC164 | 别名: HC164(省略74), SN74HC164(TI)
        #           595有锁存器可并行更新，164移位时输出会"串出来"，适合简单场景
        "HC164": "74HC164",
        "SN74HC164": "74HC164",

        # ----- LM393 双电压比较器 -----
        # 标准名: LM393 | 别名: 393(省略LM), LM293(工业级), LM2903(汽车级)
        #           293/2903是不同温度等级，引脚完全兼容
        "393": "LM393",
        "LM293": "LM393",
        "LM2903": "LM393",

        # ----- MAX232 RS232收发器 -----
        # 标准名: MAX232 | 别名: MAX232A(高速版), MAX232E(ESD保护版), SP3232(Sipex兼容)
        #           A版速度更快，E版抗静电，SP3232是第三方兼容芯片
        "MAX232A": "MAX232",
        "MAX232E": "MAX232",
        "SP3232": "MAX232",

        # ----- ULN2003 达林顿驱动阵列 -----
        # 标准名: ULN2003 | 别名: ULN2004(高输入阻抗版), ULN2803(8路版)
        #           2004输入阻抗更高(适配CMOS)，2803是8路版本(引脚不同但驱动原理相同)
        "ULN2004": "ULN2003",
        "ULN2803": "ULN2003",

        # ----- LM7805 5V线性稳压器 -----
        # 标准名: LM7805 | 别名: 7805(省略LM), LM7805CT(TO-220封装), L7805(ST), MC7805(Motorola)
        #           CT=TO-220封装，不同厂商前缀但引脚定义完全相同(入-地-出)
        "7805": "LM7805",
        "LM7805CT": "LM7805",
        "L7805": "LM7805",
        "MC7805": "LM7805",

        # ----- AMS1117-3.3 3.3V低压差稳压器 -----
        # 标准名: AMS1117-3.3 | 别名: AMS1117(默认3.3V版), AMS1117-3.3V(带V后缀)
        #           AMS1117有多个输出电压版本(1.8/2.5/3.3/5.0)，3.3V最常用，作为默认映射
        #           答：因为3.3V版本在嵌入式开发中最常用，用户输入"AMS1117"大概率指3.3V版
        "AMS1117": "AMS1117-3.3",
        "AMS1117-3.3V": "AMS1117-3.3",
        
        # ----- STM32F103C8T6 (ARM Cortex-M3 MCU) -----
        # 标准名: STM32F103C8T6 | 别名: STM32F103(省略具体型号), STM32F103C8(省略封装),
        #                           BluePill(开发板昵称)
        #           C8T6含义: C=48引脚, 8=64KB Flash, T=LQFP封装, 6=工业级温度
        #           BluePill是社区对STM32F103C8T6最小系统板的昵称，非常流行
        "STM32F103": "STM32F103C8T6",
        "STM32F103C8": "STM32F103C8T6",
        "STM32F103C8T6": "STM32F103C8T6",
        "BluePill": "STM32F103C8T6",
        "STM32F103CB": "STM32F103C8T6",
        "STM32F103CBT6": "STM32F103C8T6",
        "GD32F103C8T6": "STM32F103C8T6",
        "GD32F103": "STM32F103C8T6",
        "GD32F103C8": "STM32F103C8T6",
        "APM32F103C8T6": "STM32F103C8T6",
        "APM32F103": "STM32F103C8T6",
        
        # ----- ESP32-WROOM-32 (WiFi+BLE MCU模块) -----
        # 标准名: ESP32-WROOM-32 | 别名: ESP32(通用名), ESP32-WROOM(省略-32),
        #                              ESP32D0WDQ6(芯片裸片型号)
        #           D0WDQ6是芯片裸片型号，用户一般不会用，但数据手册中可能出现
        #           ESP32引脚34-39仅输入(无上拉电阻)，这是硬件限制
        "ESP32": "ESP32-WROOM-32",
        "ESP32-WROOM": "ESP32-WROOM-32",
        "ESP32D0WDQ6": "ESP32-WROOM-32",
        "ESP32-WROOM-32D": "ESP32-WROOM-32",
        "ESP32-WROOM-32U": "ESP32-WROOM-32",
        "ESP32-S": "ESP32-WROOM-32",
        
        # ----- ESP32-S3 (WiFi+BLE MCU, Xtensa LX7双核) -----
        # 标准名: ESP32-S3 | 别名: ESP32S3(无连字符), ESP32-S3FN8(内置Flash版),
        #                     ESP32-S3R8(内置8MB PSRAM), ESP32-S3R16V(内置16MB PSRAM),
        #                     ESP32-S3FH4R2, ESP32-S3RH2
        #           QFN56封装57引脚，45个GPIO，支持USB OTG和USB JTAG
        #           FN=内置Flash, R=内置PSRAM, H=内置Flash+PSRAM
        "ESP32S3": "ESP32-S3",
        "ESP32-S3FN8": "ESP32-S3",
        "ESP32-S3R8": "ESP32-S3",
        "ESP32-S3R16V": "ESP32-S3",
        "ESP32-S3FH4R2": "ESP32-S3",
        "ESP32-S3RH2": "ESP32-S3",
        "ESP32-S3R2": "ESP32-S3",
        "ESP32-S3R8V": "ESP32-S3",
        
        # ----- ATmega328P (AVR MCU, Arduino核心) -----
        # 标准名: ATmega328P | 别名: ATmega328(无P后缀), ATMEGA328(全大写),
        #                           ATmega328P-PU(DIP封装), ArduinoUno/Nano(开发板名)
        #           328(无P)是旧版，引脚相同但功耗略高
        #           -PU后缀表示DIP-28封装(P=Plastic, U=Tube包装)
        #           ArduinoUno/Nano映射到328P是因为用户常以板名代指芯片
        #           Uno用DIP-28版，Nano用贴片版，但引脚功能定义相同
        "ATmega328": "ATmega328P",
        "ATMEGA328": "ATmega328P",
        "ATmega328P-PU": "ATmega328P",
        "ArduinoUno": "ATmega328P",
        "ArduinoNano": "ATmega328P",
    }

    STANDARD_PINOUTS: Dict[str, Dict] = {
        # Each entry: {"pin_count": int, "package": str, "pinout": [{number, name, pin_type, functions, description}]}

        # ----- 74HC04: 六反相器（Hex Inverter） -----
        #           引脚排列规律: 1A-1Y, 2A-2Y, 3A-3Y | GND | 4Y-4A, 5Y-5A, 6Y-6A | VCC
        #           左半边1-6脚是前3个门，右半边8-13脚是后3个门，7=GND, 14=VCC
        "74HC04": {
            "pin_count": 14,
            "package": "DIP14",
            "pinout": [
                {"number": 1, "name": "1A", "pin_type": "input", "functions": ["反相器输入"], "description": "第一个反相器输入"},
                {"number": 2, "name": "1Y", "pin_type": "output", "functions": ["反相器输出"], "description": "第一个反相器输出"},
                {"number": 3, "name": "2A", "pin_type": "input", "functions": ["反相器输入"], "description": "第二个反相器输入"},
                {"number": 4, "name": "2Y", "pin_type": "output", "functions": ["反相器输出"], "description": "第二个反相器输出"},
                {"number": 5, "name": "3A", "pin_type": "input", "functions": ["反相器输入"], "description": "第三个反相器输入"},
                {"number": 6, "name": "3Y", "pin_type": "output", "functions": ["反相器输出"], "description": "第三个反相器输出"},
                {"number": 7, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 8, "name": "4Y", "pin_type": "output", "functions": ["反相器输出"], "description": "第四个反相器输出"},
                {"number": 9, "name": "4A", "pin_type": "input", "functions": ["反相器输入"], "description": "第四个反相器输入"},
                {"number": 10, "name": "5Y", "pin_type": "output", "functions": ["反相器输出"], "description": "第五个反相器输出"},
                {"number": 11, "name": "5A", "pin_type": "input", "functions": ["反相器输入"], "description": "第五个反相器输入"},
                {"number": 12, "name": "6Y", "pin_type": "output", "functions": ["反相器输出"], "description": "第六个反相器输出"},
                {"number": 13, "name": "6A", "pin_type": "input", "functions": ["反相器输入"], "description": "第六个反相器输入"},
                {"number": 14, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- 74HC595: 8位移位寄存器（带输出锁存） -----
        #           SHCP=移位时钟(数据移位), STCP=锁存时钟(数据从移位寄存器→输出寄存器)
        #           MR=主复位(清零移位寄存器), OE=输出使能(控制输出高阻/有效)
        #           工作流程: DS输入数据 → SHCP上升沿移位 → STCP上升沿锁存 → OE拉低输出
        "74HC595": {
            "pin_count": 16,
            "package": "DIP16",
            "pinout": [
                {"number": 1, "name": "Q1", "pin_type": "output", "functions": ["输出"], "description": "输出1"},
                {"number": 2, "name": "Q2", "pin_type": "output", "functions": ["输出"], "description": "输出2"},
                {"number": 3, "name": "Q3", "pin_type": "output", "functions": ["输出"], "description": "输出3"},
                {"number": 4, "name": "Q4", "pin_type": "output", "functions": ["输出"], "description": "输出4"},
                {"number": 5, "name": "Q5", "pin_type": "output", "functions": ["输出"], "description": "输出5"},
                {"number": 6, "name": "Q6", "pin_type": "output", "functions": ["输出"], "description": "输出6"},
                {"number": 7, "name": "Q7", "pin_type": "output", "functions": ["输出"], "description": "输出7"},
                {"number": 8, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 9, "name": "Q7'", "pin_type": "output", "functions": ["串行输出"], "description": "串行输出"},
                {"number": 10, "name": "MR", "pin_type": "input", "functions": ["主复位"], "description": "主复位（低电平有效）"},
                {"number": 11, "name": "SHCP", "pin_type": "input", "functions": ["移位时钟"], "description": "移位寄存器时钟输入"},
                {"number": 12, "name": "STCP", "pin_type": "input", "functions": ["锁存时钟"], "description": "存储寄存器时钟输入"},
                {"number": 13, "name": "OE", "pin_type": "input", "functions": ["输出使能"], "description": "输出使能（低电平有效）"},
                {"number": 14, "name": "DS", "pin_type": "input", "functions": ["数据输入"], "description": "串行数据输入"},
                {"number": 15, "name": "Q0", "pin_type": "output", "functions": ["输出"], "description": "输出0"},
                {"number": 16, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- 74LS00: 四2输入与非门（Quad 2-input NAND） -----
        #           与非门是"万能门"，可组合实现任何逻辑功能（与、或、非均可由NAND构成）
        "74LS00": {
            "pin_count": 14,
            "package": "DIP14",
            "pinout": [
                {"number": 1, "name": "1A", "pin_type": "input", "functions": ["与非门输入"], "description": "第一个与非门输入A"},
                {"number": 2, "name": "1B", "pin_type": "input", "functions": ["与非门输入"], "description": "第一个与非门输入B"},
                {"number": 3, "name": "1Y", "pin_type": "output", "functions": ["与非门输出"], "description": "第一个与非门输出"},
                {"number": 4, "name": "2A", "pin_type": "input", "functions": ["与非门输入"], "description": "第二个与非门输入A"},
                {"number": 5, "name": "2B", "pin_type": "input", "functions": ["与非门输入"], "description": "第二个与非门输入B"},
                {"number": 6, "name": "2Y", "pin_type": "output", "functions": ["与非门输出"], "description": "第二个与非门输出"},
                {"number": 7, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 8, "name": "3Y", "pin_type": "output", "functions": ["与非门输出"], "description": "第三个与非门输出"},
                {"number": 9, "name": "3A", "pin_type": "input", "functions": ["与非门输入"], "description": "第三个与非门输入A"},
                {"number": 10, "name": "3B", "pin_type": "input", "functions": ["与非门输入"], "description": "第三个与非门输入B"},
                {"number": 11, "name": "4Y", "pin_type": "output", "functions": ["与非门输出"], "description": "第四个与非门输出"},
                {"number": 12, "name": "4A", "pin_type": "input", "functions": ["与非门输入"], "description": "第四个与非门输入A"},
                {"number": 13, "name": "4B", "pin_type": "input", "functions": ["与非门输入"], "description": "第四个与非门输入B"},
                {"number": 14, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- NE555: 定时器（Timer） -----
        #           TRIG(2脚)<VCC/3时输出高电平, THR(6脚)>2VCC/3时输出低电平
        #           CV(5脚)接电容滤波，不接时内部参考为2VCC/3和VCC/3
        #           DIS(7脚)是开路集电极输出，用于定时电容放电
        "NE555": {
            "pin_count": 8,
            "package": "DIP8",
            "pinout": [
                {"number": 1, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 2, "name": "TRIG", "pin_type": "input", "functions": ["触发"], "description": "触发输入"},
                {"number": 3, "name": "OUT", "pin_type": "output", "functions": ["输出"], "description": "输出"},
                {"number": 4, "name": "RESET", "pin_type": "input", "functions": ["复位"], "description": "复位（低电平有效）"},
                {"number": 5, "name": "CV", "pin_type": "io", "functions": ["控制电压"], "description": "控制电压"},
                {"number": 6, "name": "THR", "pin_type": "input", "functions": ["阈值"], "description": "阈值输入"},
                {"number": 7, "name": "DIS", "pin_type": "output", "functions": ["放电"], "description": "放电"},
                {"number": 8, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- LM358: 双运算放大器（Dual Op-Amp） -----
        #           开环增益高(100dB)，但精度一般(输入失调电压±7mV)
        #           引脚规律: OUT1-IN1--IN1+ | GND | IN2+-IN2--OUT2 | VCC
        "LM358": {
            "pin_count": 8,
            "package": "DIP8",
            "pinout": [
                {"number": 1, "name": "OUT1", "pin_type": "output", "functions": ["输出"], "description": "第一个运算放大器输出"},
                {"number": 2, "name": "IN1-", "pin_type": "input", "functions": ["反相输入"], "description": "第一个运算放大器反相输入"},
                {"number": 3, "name": "IN1+", "pin_type": "input", "functions": ["同相输入"], "description": "第一个运算放大器同相输入"},
                {"number": 4, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 5, "name": "IN2+", "pin_type": "input", "functions": ["同相输入"], "description": "第二个运算放大器同相输入"},
                {"number": 6, "name": "IN2-", "pin_type": "input", "functions": ["反相输入"], "description": "第二个运算放大器反相输入"},
                {"number": 7, "name": "OUT2", "pin_type": "output", "functions": ["输出"], "description": "第二个运算放大器输出"},
                {"number": 8, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- NE5532: 双低噪声运算放大器（Dual Low-Noise Op-Amp） -----
        #           双电源供电(±3V~±20V)，4脚VEE接负电源(不是GND！)
        "NE5532": {
            "pin_count": 8,
            "package": "DIP8",
            "pinout": [
                {"number": 1, "name": "OUT1", "pin_type": "output", "functions": ["输出"], "description": "第一个运算放大器输出"},
                {"number": 2, "name": "IN1-", "pin_type": "input", "functions": ["反相输入"], "description": "第一个运算放大器反相输入"},
                {"number": 3, "name": "IN1+", "pin_type": "input", "functions": ["同相输入"], "description": "第一个运算放大器同相输入"},
                {"number": 4, "name": "VEE", "pin_type": "ground", "functions": ["负电源"], "description": "负电源（或接地）"},
                {"number": 5, "name": "IN2+", "pin_type": "input", "functions": ["同相输入"], "description": "第二个运算放大器同相输入"},
                {"number": 6, "name": "IN2-", "pin_type": "input", "functions": ["反相输入"], "description": "第二个运算放大器反相输入"},
                {"number": 7, "name": "OUT2", "pin_type": "output", "functions": ["输出"], "description": "第二个运算放大器输出"},
                {"number": 8, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "正电源"}
            ]
        },
        # ----- 74HC165: 8位并入串出移位寄存器 -----
        #           PL=并行加载(低电平有效，拉低时D0-D7数据并行载入)
        #           CE=时钟使能(低电平有效)，CP=时钟上升沿移位
        #           Q7=反相串行输出, Q7H=同相串行输出(两个输出方便不同逻辑需求)
        "74HC165": {
            "pin_count": 16,
            "package": "DIP16",
            "pinout": [
                {"number": 1, "name": "PL", "pin_type": "input", "functions": ["并行加载"], "description": "并行加载使能（低电平有效）"},
                {"number": 2, "name": "CP", "pin_type": "input", "functions": ["时钟"], "description": "时钟输入"},
                {"number": 3, "name": "D4", "pin_type": "input", "functions": ["并行输入"], "description": "并行数据输入4"},
                {"number": 4, "name": "D5", "pin_type": "input", "functions": ["并行输入"], "description": "并行数据输入5"},
                {"number": 5, "name": "D6", "pin_type": "input", "functions": ["并行输入"], "description": "并行数据输入6"},
                {"number": 6, "name": "D7", "pin_type": "input", "functions": ["并行输入"], "description": "并行数据输入7"},
                {"number": 7, "name": "Q7", "pin_type": "output", "functions": ["串行输出"], "description": "串行输出（反相）"},
                {"number": 8, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 9, "name": "Q7H", "pin_type": "output", "functions": ["串行输出"], "description": "串行输出（同相）"},
                {"number": 10, "name": "DS", "pin_type": "input", "functions": ["串行输入"], "description": "串行数据输入"},
                {"number": 11, "name": "D0", "pin_type": "input", "functions": ["并行输入"], "description": "并行数据输入0"},
                {"number": 12, "name": "D1", "pin_type": "input", "functions": ["并行输入"], "description": "并行数据输入1"},
                {"number": 13, "name": "D2", "pin_type": "input", "functions": ["并行输入"], "description": "并行数据输入2"},
                {"number": 14, "name": "D3", "pin_type": "input", "functions": ["并行输入"], "description": "并行数据输入3"},
                {"number": 15, "name": "CE", "pin_type": "input", "functions": ["时钟使能"], "description": "时钟使能（低电平有效）"},
                {"number": 16, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- 74HC245: 八路双向总线收发器 -----
        #           OE=输出使能(低电平有效)，OE=1时所有输出高阻态(总线隔离)
        #           常用于: 电平转换(5V↔3.3V)、总线驱动(增加驱动能力)、总线隔离
        #           A1-A8和B1-B8是对称的双向端口，pin_type="io"表示硬件双向
        "74HC245": {
            "pin_count": 20,
            "package": "DIP20",
            "pinout": [
                {"number": 1, "name": "DIR", "pin_type": "input", "functions": ["方向控制"], "description": "方向控制"},
                {"number": 2, "name": "A1", "pin_type": "io", "functions": ["数据"], "description": "A端口数据1"},
                {"number": 3, "name": "A2", "pin_type": "io", "functions": ["数据"], "description": "A端口数据2"},
                {"number": 4, "name": "A3", "pin_type": "io", "functions": ["数据"], "description": "A端口数据3"},
                {"number": 5, "name": "A4", "pin_type": "io", "functions": ["数据"], "description": "A端口数据4"},
                {"number": 6, "name": "A5", "pin_type": "io", "functions": ["数据"], "description": "A端口数据5"},
                {"number": 7, "name": "A6", "pin_type": "io", "functions": ["数据"], "description": "A端口数据6"},
                {"number": 8, "name": "A7", "pin_type": "io", "functions": ["数据"], "description": "A端口数据7"},
                {"number": 9, "name": "A8", "pin_type": "io", "functions": ["数据"], "description": "A端口数据8"},
                {"number": 10, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 11, "name": "B8", "pin_type": "io", "functions": ["数据"], "description": "B端口数据8"},
                {"number": 12, "name": "B7", "pin_type": "io", "functions": ["数据"], "description": "B端口数据7"},
                {"number": 13, "name": "B6", "pin_type": "io", "functions": ["数据"], "description": "B端口数据6"},
                {"number": 14, "name": "B5", "pin_type": "io", "functions": ["数据"], "description": "B端口数据5"},
                {"number": 15, "name": "B4", "pin_type": "io", "functions": ["数据"], "description": "B端口数据4"},
                {"number": 16, "name": "B3", "pin_type": "io", "functions": ["数据"], "description": "B端口数据3"},
                {"number": 17, "name": "B2", "pin_type": "io", "functions": ["数据"], "description": "B端口数据2"},
                {"number": 18, "name": "B1", "pin_type": "io", "functions": ["数据"], "description": "B端口数据1"},
                {"number": 19, "name": "OE", "pin_type": "input", "functions": ["输出使能"], "description": "输出使能（低电平有效）"},
                {"number": 20, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- 74HC138: 3-8线译码器/多路分配器 -----
        #           3个使能端: E1/E2低电平有效, E3高电平有效，三者全部满足才工作
        #           常用于: 内存地址译码、LED扫描、片选信号生成
        "74HC138": {
            "pin_count": 16,
            "package": "DIP16",
            "pinout": [
                {"number": 1, "name": "A0", "pin_type": "input", "functions": ["地址输入"], "description": "地址输入A0"},
                {"number": 2, "name": "A1", "pin_type": "input", "functions": ["地址输入"], "description": "地址输入A1"},
                {"number": 3, "name": "A2", "pin_type": "input", "functions": ["地址输入"], "description": "地址输入A2"},
                {"number": 4, "name": "E1", "pin_type": "input", "functions": ["使能"], "description": "使能输入1（低电平有效）"},
                {"number": 5, "name": "E2", "pin_type": "input", "functions": ["使能"], "description": "使能输入2（低电平有效）"},
                {"number": 6, "name": "E3", "pin_type": "input", "functions": ["使能"], "description": "使能输入3（高电平有效）"},
                {"number": 7, "name": "Y7", "pin_type": "output", "functions": ["输出"], "description": "输出Y7（低电平有效）"},
                {"number": 8, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 9, "name": "Y6", "pin_type": "output", "functions": ["输出"], "description": "输出Y6（低电平有效）"},
                {"number": 10, "name": "Y5", "pin_type": "output", "functions": ["输出"], "description": "输出Y5（低电平有效）"},
                {"number": 11, "name": "Y4", "pin_type": "output", "functions": ["输出"], "description": "输出Y4（低电平有效）"},
                {"number": 12, "name": "Y3", "pin_type": "output", "functions": ["输出"], "description": "输出Y3（低电平有效）"},
                {"number": 13, "name": "Y2", "pin_type": "output", "functions": ["输出"], "description": "输出Y2（低电平有效）"},
                {"number": 14, "name": "Y1", "pin_type": "output", "functions": ["输出"], "description": "输出Y1（低电平有效）"},
                {"number": 15, "name": "Y0", "pin_type": "output", "functions": ["输出"], "description": "输出Y0（低电平有效）"},
                {"number": 16, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- LM393: 双电压比较器 -----
        #           开路集电极输出——必须外接上拉电阻，否则输出始终为低
        #           引脚排列与LM358完全相同，但内部电路和应用场景不同
        "LM393": {
            "pin_count": 8,
            "package": "DIP8",
            "pinout": [
                {"number": 1, "name": "OUT1", "pin_type": "output", "functions": ["输出"], "description": "第一个比较器输出（开路集电极）"},
                {"number": 2, "name": "IN1-", "pin_type": "input", "functions": ["反相输入"], "description": "第一个比较器反相输入"},
                {"number": 3, "name": "IN1+", "pin_type": "input", "functions": ["同相输入"], "description": "第一个比较器同相输入"},
                {"number": 4, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 5, "name": "IN2+", "pin_type": "input", "functions": ["同相输入"], "description": "第二个比较器同相输入"},
                {"number": 6, "name": "IN2-", "pin_type": "input", "functions": ["反相输入"], "description": "第二个比较器反相输入"},
                {"number": 7, "name": "OUT2", "pin_type": "output", "functions": ["输出"], "description": "第二个比较器输出（开路集电极）"},
                {"number": 8, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- MAX232: RS-232收发器 -----
        #           内部电荷泵升压，只需单5V供电即可产生±10V RS-232电平
        #           需4个外接电容(1μF)用于电荷泵: C1+/C1-, C2+/C2-, VS+/VS-
        #           2路驱动器(TTL→RS232) + 2路接收器(RS232→TTL)
        "MAX232": {
            "pin_count": 16,
            "package": "DIP16",
            "pinout": [
                {"number": 1, "name": "C1+", "pin_type": "io", "functions": ["电容"], "description": "电容1正极"},
                {"number": 2, "name": "VS+", "pin_type": "output", "functions": ["电压"], "description": "倍压输出（+10V）"},
                {"number": 3, "name": "C1-", "pin_type": "io", "functions": ["电容"], "description": "电容1负极"},
                {"number": 4, "name": "C2+", "pin_type": "io", "functions": ["电容"], "description": "电容2正极"},
                {"number": 5, "name": "C2-", "pin_type": "io", "functions": ["电容"], "description": "电容2负极"},
                {"number": 6, "name": "VS-", "pin_type": "output", "functions": ["电压"], "description": "反相输出（-10V）"},
                {"number": 7, "name": "T2OUT", "pin_type": "output", "functions": ["RS232输出"], "description": "RS232驱动器2输出"},
                {"number": 8, "name": "R2IN", "pin_type": "input", "functions": ["RS232输入"], "description": "RS232接收器2输入"},
                {"number": 9, "name": "R2OUT", "pin_type": "output", "functions": ["TTL输出"], "description": "TTL接收器2输出"},
                {"number": 10, "name": "T2IN", "pin_type": "input", "functions": ["TTL输入"], "description": "TTL驱动器2输入"},
                {"number": 11, "name": "T1IN", "pin_type": "input", "functions": ["TTL输入"], "description": "TTL驱动器1输入"},
                {"number": 12, "name": "R1OUT", "pin_type": "output", "functions": ["TTL输出"], "description": "TTL接收器1输出"},
                {"number": 13, "name": "R1IN", "pin_type": "input", "functions": ["RS232输入"], "description": "RS232接收器1输入"},
                {"number": 14, "name": "T1OUT", "pin_type": "output", "functions": ["RS232输出"], "description": "RS232驱动器1输出"},
                {"number": 15, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 16, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极（5V）"}
            ]
        },
        # ----- ULN2003: 7路达林顿驱动阵列 -----
        #           COM(9脚)接感性负载电源，内置续流二极管保护
        #           输入1-7对应输出16-10(反向排列！IN1→OUT1即16脚)
        #           开路集电极输出——负载接在VCC和OUT之间
        "ULN2003": {
            "pin_count": 16,
            "package": "DIP16",
            "pinout": [
                {"number": 1, "name": "IN1", "pin_type": "input", "functions": ["输入"], "description": "驱动器1输入"},
                {"number": 2, "name": "IN2", "pin_type": "input", "functions": ["输入"], "description": "驱动器2输入"},
                {"number": 3, "name": "IN3", "pin_type": "input", "functions": ["输入"], "description": "驱动器3输入"},
                {"number": 4, "name": "IN4", "pin_type": "input", "functions": ["输入"], "description": "驱动器4输入"},
                {"number": 5, "name": "IN5", "pin_type": "input", "functions": ["输入"], "description": "驱动器5输入"},
                {"number": 6, "name": "IN6", "pin_type": "input", "functions": ["输入"], "description": "驱动器6输入"},
                {"number": 7, "name": "IN7", "pin_type": "input", "functions": ["输入"], "description": "驱动器7输入"},
                {"number": 8, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 9, "name": "COM", "pin_type": "power", "functions": ["公共端"], "description": "公共端（接感性负载电源）"},
                {"number": 10, "name": "OUT7", "pin_type": "output", "functions": ["输出"], "description": "驱动器7输出（开路集电极）"},
                {"number": 11, "name": "OUT6", "pin_type": "output", "functions": ["输出"], "description": "驱动器6输出（开路集电极）"},
                {"number": 12, "name": "OUT5", "pin_type": "output", "functions": ["输出"], "description": "驱动器5输出（开路集电极）"},
                {"number": 13, "name": "OUT4", "pin_type": "output", "functions": ["输出"], "description": "驱动器4输出（开路集电极）"},
                {"number": 14, "name": "OUT3", "pin_type": "output", "functions": ["输出"], "description": "驱动器3输出（开路集电极）"},
                {"number": 15, "name": "OUT2", "pin_type": "output", "functions": ["输出"], "description": "驱动器2输出（开路集电极）"},
                {"number": 16, "name": "OUT1", "pin_type": "output", "functions": ["输出"], "description": "驱动器1输出（开路集电极）"}
            ]
        },
        # ----- LM7805: 5V正电压线性稳压器 -----
        #           输入7-35V，输出5V±2%，最大1A(需散热片)
        #           TO-220封装3引脚: IN-GND-OUT(从左到右正面朝自己)
        #           78系列: 7805(5V), 7812(12V), 7824(24V)；79系列为负电压版
        "LM7805": {
            "pin_count": 3,
            "package": "TO-220",
            "pinout": [
                {"number": 1, "name": "VIN", "pin_type": "power", "functions": ["输入"], "description": "电压输入（7-35V）"},
                {"number": 2, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 3, "name": "VOUT", "pin_type": "power", "functions": ["输出"], "description": "稳压输出（5V）"}
            ]
        },
        # ----- AMS1117-3.3: 3.3V低压差线性稳压器(LDO) -----
        #           SOT-223封装4引脚: 1=GND, 2=VOUT, 3=VIN, 4=VOUT(散热片)
        #           2脚和4脚都是VOUT，4脚兼作散热焊盘，增大散热面积
        #           最大输出800mA，常用于3.3V MCU供电
        "AMS1117-3.3": {
            "pin_count": 4,
            "package": "SOT-223",
            "pinout": [
                {"number": 1, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地/调整端"},
                {"number": 2, "name": "VOUT", "pin_type": "power", "functions": ["输出"], "description": "稳压输出（3.3V）"},
                {"number": 3, "name": "VIN", "pin_type": "power", "functions": ["输入"], "description": "电压输入（4.3-15V）"},
                {"number": 4, "name": "VOUT", "pin_type": "power", "functions": ["输出"], "description": "稳压输出（散热片）"}
            ]
        },
        # ----- 74HC08: 四2输入与门（Quad 2-input AND） -----
        "74HC08": {
            "pin_count": 14,
            "package": "DIP14",
            "pinout": [
                {"number": 1, "name": "1A", "pin_type": "input", "functions": ["与门输入"], "description": "第一个与门输入A"},
                {"number": 2, "name": "1B", "pin_type": "input", "functions": ["与门输入"], "description": "第一个与门输入B"},
                {"number": 3, "name": "1Y", "pin_type": "output", "functions": ["与门输出"], "description": "第一个与门输出"},
                {"number": 4, "name": "2A", "pin_type": "input", "functions": ["与门输入"], "description": "第二个与门输入A"},
                {"number": 5, "name": "2B", "pin_type": "input", "functions": ["与门输入"], "description": "第二个与门输入B"},
                {"number": 6, "name": "2Y", "pin_type": "output", "functions": ["与门输出"], "description": "第二个与门输出"},
                {"number": 7, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 8, "name": "3Y", "pin_type": "output", "functions": ["与门输出"], "description": "第三个与门输出"},
                {"number": 9, "name": "3A", "pin_type": "input", "functions": ["与门输入"], "description": "第三个与门输入A"},
                {"number": 10, "name": "3B", "pin_type": "input", "functions": ["与门输入"], "description": "第三个与门输入B"},
                {"number": 11, "name": "4Y", "pin_type": "output", "functions": ["与门输出"], "description": "第四个与门输出"},
                {"number": 12, "name": "4A", "pin_type": "input", "functions": ["与门输入"], "description": "第四个与门输入A"},
                {"number": 13, "name": "4B", "pin_type": "input", "functions": ["与门输入"], "description": "第四个与门输入B"},
                {"number": 14, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- 74HC02: 四2输入或非门（Quad 2-input NOR） -----
        #           02的输出在1脚(1Y)，而00/08/32的输出在3脚(1Y)
        "74HC02": {
            "pin_count": 14,
            "package": "DIP14",
            "pinout": [
                {"number": 1, "name": "1Y", "pin_type": "output", "functions": ["或非门输出"], "description": "第一个或非门输出"},
                {"number": 2, "name": "1A", "pin_type": "input", "functions": ["或非门输入"], "description": "第一个或非门输入A"},
                {"number": 3, "name": "1B", "pin_type": "input", "functions": ["或非门输入"], "description": "第一个或非门输入B"},
                {"number": 4, "name": "2Y", "pin_type": "output", "functions": ["或非门输出"], "description": "第二个或非门输出"},
                {"number": 5, "name": "2A", "pin_type": "input", "functions": ["或非门输入"], "description": "第二个或非门输入A"},
                {"number": 6, "name": "2B", "pin_type": "input", "functions": ["或非门输入"], "description": "第二个或非门输入B"},
                {"number": 7, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 8, "name": "3A", "pin_type": "input", "functions": ["或非门输入"], "description": "第三个或非门输入A"},
                {"number": 9, "name": "3B", "pin_type": "input", "functions": ["或非门输入"], "description": "第三个或非门输入B"},
                {"number": 10, "name": "3Y", "pin_type": "output", "functions": ["或非门输出"], "description": "第三个或非门输出"},
                {"number": 11, "name": "4A", "pin_type": "input", "functions": ["或非门输入"], "description": "第四个或非门输入A"},
                {"number": 12, "name": "4B", "pin_type": "input", "functions": ["或非门输入"], "description": "第四个或非门输入B"},
                {"number": 13, "name": "4Y", "pin_type": "output", "functions": ["或非门输出"], "description": "第四个或非门输出"},
                {"number": 14, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- 74HC32: 四2输入或门（Quad 2-input OR） -----
        "74HC32": {
            "pin_count": 14,
            "package": "DIP14",
            "pinout": [
                {"number": 1, "name": "1A", "pin_type": "input", "functions": ["或门输入"], "description": "第一个或门输入A"},
                {"number": 2, "name": "1B", "pin_type": "input", "functions": ["或门输入"], "description": "第一个或门输入B"},
                {"number": 3, "name": "1Y", "pin_type": "output", "functions": ["或门输出"], "description": "第一个或门输出"},
                {"number": 4, "name": "2A", "pin_type": "input", "functions": ["或门输入"], "description": "第二个或门输入A"},
                {"number": 5, "name": "2B", "pin_type": "input", "functions": ["或门输入"], "description": "第二个或门输入B"},
                {"number": 6, "name": "2Y", "pin_type": "output", "functions": ["或门输出"], "description": "第二个或门输出"},
                {"number": 7, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 8, "name": "3Y", "pin_type": "output", "functions": ["或门输出"], "description": "第三个或门输出"},
                {"number": 9, "name": "3A", "pin_type": "input", "functions": ["或门输入"], "description": "第三个或门输入A"},
                {"number": 10, "name": "3B", "pin_type": "input", "functions": ["或门输入"], "description": "第三个或门输入B"},
                {"number": 11, "name": "4Y", "pin_type": "output", "functions": ["或门输出"], "description": "第四个或门输出"},
                {"number": 12, "name": "4A", "pin_type": "input", "functions": ["或门输入"], "description": "第四个或门输入A"},
                {"number": 13, "name": "4B", "pin_type": "input", "functions": ["或门输入"], "description": "第四个或门输入B"},
                {"number": 14, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- 74HC126: 四路三态缓冲器（Quad 3-State Buffer） -----
        #           三态输出允许多个芯片共享同一总线(同一时刻只有一个OE有效)
        #           126是高电平使能，125是低电平使能，两者互补
        "74HC126": {
            "pin_count": 14,
            "package": "DIP14",
            "pinout": [
                {"number": 1, "name": "1OE", "pin_type": "input", "functions": ["输出使能"], "description": "第一个缓冲器输出使能"},
                {"number": 2, "name": "1A", "pin_type": "input", "functions": ["输入"], "description": "第一个缓冲器输入"},
                {"number": 3, "name": "1Y", "pin_type": "output", "functions": ["输出"], "description": "第一个缓冲器输出"},
                {"number": 4, "name": "2OE", "pin_type": "input", "functions": ["输出使能"], "description": "第二个缓冲器输出使能"},
                {"number": 5, "name": "2A", "pin_type": "input", "functions": ["输入"], "description": "第二个缓冲器输入"},
                {"number": 6, "name": "2Y", "pin_type": "output", "functions": ["输出"], "description": "第二个缓冲器输出"},
                {"number": 7, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 8, "name": "3Y", "pin_type": "output", "functions": ["输出"], "description": "第三个缓冲器输出"},
                {"number": 9, "name": "3A", "pin_type": "input", "functions": ["输入"], "description": "第三个缓冲器输入"},
                {"number": 10, "name": "3OE", "pin_type": "input", "functions": ["输出使能"], "description": "第三个缓冲器输出使能"},
                {"number": 11, "name": "4Y", "pin_type": "output", "functions": ["输出"], "description": "第四个缓冲器输出"},
                {"number": 12, "name": "4A", "pin_type": "input", "functions": ["输入"], "description": "第四个缓冲器输入"},
                {"number": 13, "name": "4OE", "pin_type": "input", "functions": ["输出使能"], "description": "第四个缓冲器输出使能"},
                {"number": 14, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- 74HC164: 8位串入并出移位寄存器 -----
        #           DSA和DSB是两个串行输入，两者相与作为实际输入(DSA AND DSB)
        #           可用作简单输入使能：一个接数据，一个接使能信号
        #           MR=低电平清零所有输出，CP=时钟上升沿移位
        "74HC164": {
            "pin_count": 14,
            "package": "DIP14",
            "pinout": [
                {"number": 1, "name": "DSA", "pin_type": "input", "functions": ["串行输入"], "description": "串行数据输入A"},
                {"number": 2, "name": "DSB", "pin_type": "input", "functions": ["串行输入"], "description": "串行数据输入B"},
                {"number": 3, "name": "Q0", "pin_type": "output", "functions": ["输出"], "description": "并行输出0"},
                {"number": 4, "name": "Q1", "pin_type": "output", "functions": ["输出"], "description": "并行输出1"},
                {"number": 5, "name": "Q2", "pin_type": "output", "functions": ["输出"], "description": "并行输出2"},
                {"number": 6, "name": "Q3", "pin_type": "output", "functions": ["输出"], "description": "并行输出3"},
                {"number": 7, "name": "GND", "pin_type": "ground", "functions": ["接地"], "description": "接地"},
                {"number": 8, "name": "CP", "pin_type": "input", "functions": ["时钟"], "description": "时钟输入"},
                {"number": 9, "name": "MR", "pin_type": "input", "functions": ["主复位"], "description": "主复位（低电平有效）"},
                {"number": 10, "name": "Q4", "pin_type": "output", "functions": ["输出"], "description": "并行输出4"},
                {"number": 11, "name": "Q5", "pin_type": "output", "functions": ["输出"], "description": "并行输出5"},
                {"number": 12, "name": "Q6", "pin_type": "output", "functions": ["输出"], "description": "并行输出6"},
                {"number": 13, "name": "Q7", "pin_type": "output", "functions": ["输出"], "description": "并行输出7"},
                {"number": 14, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "电源正极"}
            ]
        },
        # ----- STM32F103C8T6: ARM Cortex-M3 微控制器 -----
        #           芯片分组:
        #           - 电源: VBAT(1), VDDA(8), VDD(14,24,36,48), VSSA(7), VSS(13,23,35,47)
        #           - 晶振: PD0-OSC_IN(38), PD1-OSC_OUT(39) [注: 部分版本无PD0/PD1]
        #           - 复位: NRST(44), BOOT0(60→实际为Pin44附近), BOOT1(21=PB2)
        #           - SWD调试: PA13-SWDIO(44), PA14-SWCLK(45)
        #           - GPIO端口: PA(9-18,40-46), PB(19-22,26-33,50-55), PC(2-6), PD(28-34)
        #           - pin_type="bidirectional"表示GPIO，可软件配置为输入/输出/复用
        #           functions字段包含所有复用功能(默认功能+复用功能+GPIO)
        "STM32F103C8T6": {
            "pin_count": 48,
            "package": "LQFP48",
            "note": "完整48引脚定义(LQFP48封装)",
            "pinout": [
                {"number": 1, "name": "VBAT", "pin_type": "power", "functions": ["VBAT"], "description": "RTC备份域电源/电池供电"},
                {"number": 2, "name": "PC13", "pin_type": "bidirectional", "functions": ["PC13", "RTC_TAMP1", "RTC_TS", "RTC_OUT", "GPIO"], "description": "RTC tamper/时间戳/事件输出"},
                {"number": 3, "name": "PC0", "pin_type": "bidirectional", "functions": ["PC0", "ADC123_IN10", "GPIO"], "description": "ADC通道10"},
                {"number": 4, "name": "PC1", "pin_type": "bidirectional", "functions": ["PC1", "ADC123_IN11", "GPIO"], "description": "ADC通道11"},
                {"number": 5, "name": "PC2", "pin_type": "bidirectional", "functions": ["PC2", "ADC123_IN12", "GPIO"], "description": "ADC通道12"},
                {"number": 6, "name": "PC3", "pin_type": "bidirectional", "functions": ["PC3", "ADC123_IN13", "GPIO"], "description": "ADC通道13"},
                {"number": 7, "name": "VSSA", "pin_type": "ground", "functions": ["VSSA"], "description": "模拟地"},
                {"number": 8, "name": "VDDA", "pin_type": "power", "functions": ["VDDA"], "description": "模拟电源"},
                {"number": 9, "name": "PA0", "pin_type": "bidirectional", "functions": ["PA0", "WKUP", "USART2_CTS", "ADC123_IN0", "TIM8_CH1", "TIM2_CH1_ETR", "GPIO"], "description": "唤醒/USART2_CTS/ADC通道0/TIM通道"},
                {"number": 10, "name": "PA1", "pin_type": "bidirectional", "functions": ["PA1", "USART2_RTS", "ADC123_IN1", "TIM8_CH2", "TIM2_CH2", "GPIO"], "description": "USART2_RTS/ADC通道1/TIM通道"},
                {"number": 11, "name": "PA2", "pin_type": "bidirectional", "functions": ["PA2", "USART2_TX", "ADC123_IN2", "TIM8_CH3", "TIM2_CH3", "GPIO"], "description": "USART2发送/ADC通道2/TIM通道"},
                {"number": 12, "name": "PA3", "pin_type": "bidirectional", "functions": ["PA3", "USART2_RX", "ADC123_IN3", "TIM8_CH4", "TIM2_CH4", "GPIO"], "description": "USART2接收/ADC通道3/TIM通道"},
                {"number": 13, "name": "VSS", "pin_type": "ground", "functions": ["VSS"], "description": "数字地"},
                {"number": 14, "name": "VDD", "pin_type": "power", "functions": ["VDD"], "description": "数字电源"},
                {"number": 15, "name": "PA4", "pin_type": "bidirectional", "functions": ["PA4", "SPI1_NSS", "USART2_CK", "ADC12_IN4", "GPIO"], "description": "SPI1片选/USART2时钟/ADC通道4"},
                {"number": 16, "name": "PA5", "pin_type": "bidirectional", "functions": ["PA5", "SPI1_SCK", "ADC12_IN5", "GPIO"], "description": "SPI1时钟/ADC通道5"},
                {"number": 17, "name": "PA6", "pin_type": "bidirectional", "functions": ["PA6", "SPI1_MISO", "ADC12_IN6", "TIM8_CH1", "TIM3_CH1", "GPIO"], "description": "SPI1主入从出/ADC通道6/TIM通道"},
                {"number": 18, "name": "PA7", "pin_type": "bidirectional", "functions": ["PA7", "SPI1_MOSI", "ADC12_IN7", "TIM8_CH2", "TIM3_CH2", "GPIO"], "description": "SPI1主出从入/ADC通道7/TIM通道"},
                {"number": 19, "name": "PB0", "pin_type": "bidirectional", "functions": ["PB0", "ADC12_IN8", "TIM8_CH2N", "TIM3_CH3", "GPIO"], "description": "ADC通道8/TIM通道互补"},
                {"number": 20, "name": "PB1", "pin_type": "bidirectional", "functions": ["PB1", "ADC12_IN9", "TIM8_CH3N", "TIM3_CH4", "GPIO"], "description": "ADC通道9/TIM通道互补"},
                {"number": 21, "name": "PB2", "pin_type": "input", "functions": ["PB2", "BOOT1", "GPIO"], "description": "启动模式选择1"},
                {"number": 22, "name": "PB10", "pin_type": "bidirectional", "functions": ["PB10", "I2C2_SCL", "USART3_TX", "TIM2_CH3", "GPIO"], "description": "I2C2时钟/USART3发送/TIM通道"},
                {"number": 23, "name": "PB11", "pin_type": "bidirectional", "functions": ["PB11", "I2C2_SDA", "USART3_RX", "TIM2_CH4", "GPIO"], "description": "I2C2数据/USART3接收/TIM通道"},
                {"number": 24, "name": "VSS_1", "pin_type": "ground", "functions": ["VSS"], "description": "数字地"},
                {"number": 25, "name": "VDD_1", "pin_type": "power", "functions": ["VDD"], "description": "数字电源"},
                {"number": 26, "name": "PB12", "pin_type": "bidirectional", "functions": ["PB12", "SPI2_NSS", "I2C2_SMBA", "USART3_CK", "TIM1_BKIN", "GPIO"], "description": "SPI2片选/I2C2_SMBA/USART3时钟/TIM1刹车"},
                {"number": 27, "name": "PB13", "pin_type": "bidirectional", "functions": ["PB13", "SPI2_SCK", "I2C2_SCL", "USART3_CTS", "TIM1_CH1N", "GPIO"], "description": "SPI2时钟/I2C2时钟/USART3_CTS/TIM1互补"},
                {"number": 28, "name": "PB14", "pin_type": "bidirectional", "functions": ["PB14", "SPI2_MISO", "I2C2_SDA", "USART3_RTS", "TIM1_CH2N", "GPIO"], "description": "SPI2主入从出/I2C2数据/USART3_RTS/TIM1互补"},
                {"number": 29, "name": "PB15", "pin_type": "bidirectional", "functions": ["PB15", "SPI2_MOSI", "TIM1_CH3N", "GPIO"], "description": "SPI2主出从入/TIM1互补通道"},
                {"number": 30, "name": "PD8", "pin_type": "bidirectional", "functions": ["PD8", "USART3_TX", "GPIO"], "description": "USART3发送"},
                {"number": 31, "name": "PD9", "pin_type": "bidirectional", "functions": ["PD9", "USART3_RX", "GPIO"], "description": "USART3接收"},
                {"number": 32, "name": "PD10", "pin_type": "bidirectional", "functions": ["PD10", "USART3_CK", "GPIO"], "description": "USART3时钟"},
                {"number": 33, "name": "PD11", "pin_type": "bidirectional", "functions": ["PD11", "USART3_CTS", "CAN_RX", "GPIO"], "description": "USART3_CTS/CAN接收"},
                {"number": 34, "name": "PD12", "pin_type": "bidirectional", "functions": ["PD12", "USART3_RTS", "TIM4_CH1", "CAN_TX", "GPIO"], "description": "USART3_RTS/TIM4通道/CAN发送"},
                {"number": 35, "name": "PD13", "pin_type": "bidirectional", "functions": ["PD13", "TIM4_CH2", "GPIO"], "description": "TIM4通道2"},
                {"number": 36, "name": "NRST", "pin_type": "input", "functions": ["NRST"], "description": "系统复位(低电平有效)"},
                {"number": 37, "name": "VSS_2", "pin_type": "ground", "functions": ["VSS"], "description": "数字地"},
                {"number": 38, "name": "OSC_IN", "pin_type": "input", "functions": ["OSC_IN", "PD0", "OSC"], "description": "晶振输入(HSE外部高速时钟)"},
                {"number": 39, "name": "OSC_OUT", "pin_type": "output", "functions": ["OSC_OUT", "PD1", "OSC"], "description": "晶振输出"},
                {"number": 40, "name": "PA8", "pin_type": "bidirectional", "functions": ["PA8", "USART1_CK", "MCO", "TIM1_CH1", "GPIO"], "description": "USART1时钟/MCO输出/TIM1通道1"},
                {"number": 41, "name": "PA9", "pin_type": "bidirectional", "functions": ["PA9", "USART1_TX", "TIM1_CH2", "CAN_TX", "GPIO"], "description": "USART1发送/TIM1通道2/CAN发送"},
                {"number": 42, "name": "PA10", "pin_type": "bidirectional", "functions": ["PA10", "USART1_RX", "TIM1_CH3", "CAN_RX", "GPIO"], "description": "USART1接收/TIM1通道3/CAN接收"},
                {"number": 43, "name": "PA11", "pin_type": "bidirectional", "functions": ["PA11", "USART1_CTS", "CAN_RX", "TIM1_CH4", "USBDM", "GPIO"], "description": "USART1_CTS/CAN接收/TIM1通道4/USB D-"},
                {"number": 44, "name": "PA12", "pin_type": "bidirectional", "functions": ["PA12", "USART1_RTS", "CAN_TX", "TIM1_ETR", "USBDP", "GPIO"], "description": "USART1_RTS/CAN发送/TIM1_ETR/USB D+"},
                {"number": 45, "name": "PA13", "pin_type": "bidirectional", "functions": ["PA13", "SWDIO", "JTMS", "SWDAT", "JTAG_TMS", "GPIO"], "description": "SWD数据线(JTAG TMS)"},
                {"number": 46, "name": "PA14", "pin_type": "bidirectional", "functions": ["PA14", "SWCLK", "JTCK", "SWCLK", "JTAG_CLK", "GPIO"], "description": "SWD时钟线(JTAG CLK)"},
                {"number": 47, "name": "PA15", "pin_type": "bidirectional", "functions": ["PA15", "SPI1_NSS", "JTDI", "TIM2_CH1_ETR", "GPIO"], "description": "SPI1片选/JTAG TDI/TIM2外部触发"},
                {"number": 48, "name": "PB3", "pin_type": "bidirectional", "functions": ["PB3", "SPI1_SCK", "JTDO", "TRACESWO", "TIM2_CH2", "GPIO"], "description": "SPI1时钟/JTAG TDO/跟踪SWO"}
            ]
        },
        # ----- ESP32-WROOM-32: WiFi+BLE MCU模块 -----
        #           仅收录15个常用关键引脚(完整38引脚需从知识图谱获取)
        #           注意硬件限制: GPIO34-39仅输入模式(无上拉/下拉电阻)
        #           GPIO0是启动模式选择: 上拉=正常启动，下拉=下载模式
        #           EN=芯片使能(低电平复位)，必须上拉到3.3V
        "ESP32-WROOM-32": {
            "pin_count": 15,
            "package": "Module",
            "note": "仅包含常用关键引脚",
            "pinout": [
                {"number": 1, "name": "GND", "pin_type": "power", "functions": ["地"], "description": "地"},
                {"number": 2, "name": "TX0", "pin_type": "output", "functions": ["UART0_TX", "GPIO1"], "description": "UART0发送"},
                {"number": 3, "name": "RX0", "pin_type": "input", "functions": ["UART0_RX", "GPIO3"], "description": "UART0接收"},
                {"number": 6, "name": "EN", "pin_type": "input", "functions": ["使能"], "description": "芯片使能"},
                {"number": 10, "name": "D0", "pin_type": "output", "functions": ["GPIO0", "Boot"], "description": "启动模式选择"},
                {"number": 19, "name": "MISO", "pin_type": "input", "functions": ["SPI_MISO", "GPIO19"], "description": "SPI主入从出"},
                {"number": 21, "name": "SDA", "pin_type": "bidirectional", "functions": ["I2C_SDA", "GPIO21"], "description": "I2C数据"},
                {"number": 22, "name": "SCL", "pin_type": "bidirectional", "functions": ["I2C_SCL", "GPIO22"], "description": "I2C时钟"},
                {"number": 23, "name": "MOSI", "pin_type": "output", "functions": ["SPI_MOSI", "GPIO23"], "description": "SPI主出从入"},
                {"number": 18, "name": "SCK", "pin_type": "output", "functions": ["SPI_SCK", "GPIO18"], "description": "SPI时钟"},
                {"number": 25, "name": "DAC1", "pin_type": "output", "functions": ["DAC1", "GPIO25"], "description": "数模转换1"},
                {"number": 26, "name": "DAC2", "pin_type": "output", "functions": ["DAC2", "GPIO26"], "description": "数模转换2"},
                {"number": 34, "name": "VN", "pin_type": "input", "functions": ["GPIO34", "ADC"], "description": "仅输入GPIO"},
                {"number": 35, "name": "VP", "pin_type": "input", "functions": ["GPIO35", "ADC"], "description": "仅输入GPIO"},
                {"number": 38, "name": "3V3", "pin_type": "power", "functions": ["电源"], "description": "3.3V电源"}
            ]
        },
        # ----- ESP32-S3: WiFi+BLE MCU (Xtensa LX7双核) -----
        #           与ESP32(经典版)的关键区别:
        #           - 无DAC(ESP32有2路DAC)
        #           - 新增USB OTG和USB Serial/JTAG
        #           - GPIO26-32用于SPI Flash/PSRAM通信，不建议用作其他用途
        #           - GPIO0/3/19/20/45/46为Strapping/受限引脚
        #           引脚分组:
        #           - 模拟: LNA_IN(1), CHIP_PU(4), XTAL_N(53), XTAL_P(54)
        #           - 电源: VDD3P3(2,3), VDD3P3_RTC(20), VDD_SPI(29), VDD3P3_CPU(46), VDDA(55,56), GND(57)
        #           - GPIO: GPIO0-GPIO21, GPIO33-GPIO46 (共45个)
        #           - SPI Flash: SPICS1(28), SPIHD(30), SPIWP(31), SPICS0(32), SPICLK(33), SPIQ(34), SPID(35), SPICLK_N(36), SPICLK_P(37)
        #           - JTAG: MTCK(44), MTDO(45), MTDI(47), MTMS(48)
        #           - UART: U0TXD(49), U0RXD(50)
        #           - 晶振: XTAL_32K_P(21), XTAL_32K_N(22)
        "ESP32-S3": {
            "pin_count": 57,
            "package": "QFN56",
            "note": "完整57引脚定义(QFN56 7x7mm封装)",
            "pinout": [
                {"number": 1, "name": "LNA_IN", "pin_type": "analog", "functions": ["RF", "LNA"], "description": "射频低噪声放大器输入"},
                {"number": 2, "name": "VDD3P3", "pin_type": "power", "functions": ["VDD3P3"], "description": "3.3V电源"},
                {"number": 3, "name": "VDD3P3", "pin_type": "power", "functions": ["VDD3P3"], "description": "3.3V电源"},
                {"number": 4, "name": "CHIP_PU", "pin_type": "input", "functions": ["CHIP_PU", "RESET"], "description": "芯片使能/复位(低电平有效)"},
                {"number": 5, "name": "GPIO0", "pin_type": "bidirectional", "functions": ["GPIO0", "BOOT_MODE", "RTC"], "description": "启动模式选择Strapping引脚"},
                {"number": 6, "name": "GPIO1", "pin_type": "bidirectional", "functions": ["GPIO1", "RTC", "ADC1_CH0"], "description": "通用GPIO/ADC通道0"},
                {"number": 7, "name": "GPIO2", "pin_type": "bidirectional", "functions": ["GPIO2", "RTC", "ADC1_CH1"], "description": "通用GPIO/ADC通道1"},
                {"number": 8, "name": "GPIO3", "pin_type": "bidirectional", "functions": ["GPIO3", "JTAG_SRC", "RTC", "ADC1_CH2"], "description": "JTAG信号源Strapping引脚"},
                {"number": 9, "name": "GPIO4", "pin_type": "bidirectional", "functions": ["GPIO4", "RTC", "ADC1_CH3"], "description": "通用GPIO/ADC通道3"},
                {"number": 10, "name": "GPIO5", "pin_type": "bidirectional", "functions": ["GPIO5", "RTC", "ADC1_CH4"], "description": "通用GPIO/ADC通道4"},
                {"number": 11, "name": "GPIO6", "pin_type": "bidirectional", "functions": ["GPIO6", "RTC", "ADC1_CH5"], "description": "通用GPIO/ADC通道5"},
                {"number": 12, "name": "GPIO7", "pin_type": "bidirectional", "functions": ["GPIO7", "RTC", "ADC1_CH6"], "description": "通用GPIO/ADC通道6"},
                {"number": 13, "name": "GPIO8", "pin_type": "bidirectional", "functions": ["GPIO8", "RTC", "ADC1_CH7"], "description": "通用GPIO/ADC通道7"},
                {"number": 14, "name": "GPIO9", "pin_type": "bidirectional", "functions": ["GPIO9", "RTC", "ADC1_CH8", "TOUCH"], "description": "通用GPIO/ADC/触摸"},
                {"number": 15, "name": "GPIO10", "pin_type": "bidirectional", "functions": ["GPIO10", "RTC", "ADC1_CH9", "TOUCH"], "description": "通用GPIO/ADC/触摸"},
                {"number": 16, "name": "GPIO11", "pin_type": "bidirectional", "functions": ["GPIO11", "RTC", "ADC2_CH0", "TOUCH"], "description": "通用GPIO/ADC/触摸"},
                {"number": 17, "name": "GPIO12", "pin_type": "bidirectional", "functions": ["GPIO12", "RTC", "ADC2_CH1", "TOUCH"], "description": "通用GPIO/ADC/触摸"},
                {"number": 18, "name": "GPIO13", "pin_type": "bidirectional", "functions": ["GPIO13", "RTC", "ADC2_CH2", "TOUCH"], "description": "通用GPIO/ADC/触摸"},
                {"number": 19, "name": "GPIO14", "pin_type": "bidirectional", "functions": ["GPIO14", "RTC", "ADC2_CH3", "TOUCH"], "description": "通用GPIO/ADC/触摸"},
                {"number": 20, "name": "VDD3P3_RTC", "pin_type": "power", "functions": ["VDD3P3_RTC"], "description": "RTC域3.3V电源"},
                {"number": 21, "name": "XTAL_32K_P", "pin_type": "bidirectional", "functions": ["XTAL_32K_P", "GPIO17", "RTC"], "description": "32K晶振正端/GPIO17"},
                {"number": 22, "name": "XTAL_32K_N", "pin_type": "bidirectional", "functions": ["XTAL_32K_N", "GPIO18", "RTC"], "description": "32K晶振负端/GPIO18"},
                {"number": 23, "name": "GPIO17", "pin_type": "bidirectional", "functions": ["GPIO17", "RTC"], "description": "通用GPIO"},
                {"number": 24, "name": "GPIO18", "pin_type": "bidirectional", "functions": ["GPIO18", "RTC", "USB_PU"], "description": "通用GPIO/USB上拉"},
                {"number": 25, "name": "GPIO19", "pin_type": "bidirectional", "functions": ["GPIO19", "U0TXD", "RTC"], "description": "UART0发送(调试用)"},
                {"number": 26, "name": "GPIO20", "pin_type": "bidirectional", "functions": ["GPIO20", "U0RXD", "RTC"], "description": "UART0接收(调试用)"},
                {"number": 27, "name": "GPIO21", "pin_type": "bidirectional", "functions": ["GPIO21", "RTC"], "description": "通用GPIO"},
                {"number": 28, "name": "SPICS1", "pin_type": "bidirectional", "functions": ["SPICS1", "GPIO26"], "description": "SPI Flash片选1(不建议用作GPIO)"},
                {"number": 29, "name": "VDD_SPI", "pin_type": "power", "functions": ["VDD_SPI"], "description": "SPI Flash电源(3.3V或1.8V)"},
                {"number": 30, "name": "SPIHD", "pin_type": "bidirectional", "functions": ["SPIHD", "GPIO27"], "description": "SPI Flash保持(不建议用作GPIO)"},
                {"number": 31, "name": "SPIWP", "pin_type": "bidirectional", "functions": ["SPIWP", "GPIO28"], "description": "SPI Flash写保护(不建议用作GPIO)"},
                {"number": 32, "name": "SPICS0", "pin_type": "bidirectional", "functions": ["SPICS0", "GPIO29"], "description": "SPI Flash片选0(不建议用作GPIO)"},
                {"number": 33, "name": "SPICLK", "pin_type": "bidirectional", "functions": ["SPICLK", "GPIO30"], "description": "SPI Flash时钟(不建议用作GPIO)"},
                {"number": 34, "name": "SPIQ", "pin_type": "bidirectional", "functions": ["SPIQ", "GPIO31"], "description": "SPI Flash数据Q(不建议用作GPIO)"},
                {"number": 35, "name": "SPID", "pin_type": "bidirectional", "functions": ["SPID", "GPIO32"], "description": "SPI Flash数据D(不建议用作GPIO)"},
                {"number": 36, "name": "SPICLK_N", "pin_type": "bidirectional", "functions": ["SPICLK_N"], "description": "SPI Flash差分时钟负端"},
                {"number": 37, "name": "SPICLK_P", "pin_type": "bidirectional", "functions": ["SPICLK_P"], "description": "SPI Flash差分时钟正端"},
                {"number": 38, "name": "GPIO33", "pin_type": "bidirectional", "functions": ["GPIO33"], "description": "通用GPIO"},
                {"number": 39, "name": "GPIO34", "pin_type": "bidirectional", "functions": ["GPIO34"], "description": "通用GPIO"},
                {"number": 40, "name": "GPIO35", "pin_type": "bidirectional", "functions": ["GPIO35"], "description": "通用GPIO"},
                {"number": 41, "name": "GPIO36", "pin_type": "bidirectional", "functions": ["GPIO36"], "description": "通用GPIO"},
                {"number": 42, "name": "GPIO37", "pin_type": "bidirectional", "functions": ["GPIO37"], "description": "通用GPIO"},
                {"number": 43, "name": "GPIO38", "pin_type": "bidirectional", "functions": ["GPIO38"], "description": "通用GPIO"},
                {"number": 44, "name": "MTCK", "pin_type": "bidirectional", "functions": ["MTCK", "GPIO39", "JTAG"], "description": "JTAG时钟/GPIO39"},
                {"number": 45, "name": "MTDO", "pin_type": "bidirectional", "functions": ["MTDO", "GPIO40", "JTAG"], "description": "JTAG数据输出/GPIO40"},
                {"number": 46, "name": "VDD3P3_CPU", "pin_type": "power", "functions": ["VDD3P3_CPU"], "description": "CPU域3.3V电源"},
                {"number": 47, "name": "MTDI", "pin_type": "bidirectional", "functions": ["MTDI", "GPIO41", "JTAG"], "description": "JTAG数据输入/GPIO41"},
                {"number": 48, "name": "MTMS", "pin_type": "bidirectional", "functions": ["MTMS", "GPIO42", "JTAG"], "description": "JTAG模式选择/GPIO42"},
                {"number": 49, "name": "U0TXD", "pin_type": "bidirectional", "functions": ["U0TXD", "GPIO43"], "description": "UART0发送/GPIO43"},
                {"number": 50, "name": "U0RXD", "pin_type": "bidirectional", "functions": ["U0RXD", "GPIO44"], "description": "UART0接收/GPIO44"},
                {"number": 51, "name": "GPIO45", "pin_type": "bidirectional", "functions": ["GPIO45", "VDD_SPI_SEL"], "description": "VDD_SPI电压选择Strapping引脚"},
                {"number": 52, "name": "GPIO46", "pin_type": "bidirectional", "functions": ["GPIO46", "ROM_MSG"], "description": "ROM日志打印控制Strapping引脚"},
                {"number": 53, "name": "XTAL_N", "pin_type": "analog", "functions": ["XTAL_N"], "description": "主晶振负端"},
                {"number": 54, "name": "XTAL_P", "pin_type": "analog", "functions": ["XTAL_P"], "description": "主晶振正端"},
                {"number": 55, "name": "VDDA", "pin_type": "power", "functions": ["VDDA"], "description": "模拟电源"},
                {"number": 56, "name": "VDDA", "pin_type": "power", "functions": ["VDDA"], "description": "模拟电源"},
                {"number": 57, "name": "GND", "pin_type": "ground", "functions": ["GND"], "description": "接地"}
            ]
        },
        # ----- ATmega328P: AVR 8位微控制器（Arduino核心） -----
        #           引脚分组:
        #           - PD0-PD7: Port D (含UART收发、外部中断、PWM、定时器)
        #           - PB0-PB7: Port B (含SPI接口、PWM、输入捕获、晶振)
        #           - PC0-PC5: Port C (含ADC0-5、I2C接口)
        #           - 电源: VCC(7), GND(8), RESET(1/PC6)
        #           Arduino数字引脚D0-D13 = PD0-PD7 + PB0-PB5
        #           Arduino模拟引脚A0-A5 = PC0-PC5
        "ATmega328P": {
            "pin_count": 25,
            "package": "DIP28",
            "note": "仅包含常用关键引脚",
            "pinout": [
                {"number": 1, "name": "PC6", "pin_type": "input", "functions": ["RESET"], "description": "复位"},
                {"number": 2, "name": "PD0", "pin_type": "bidirectional", "functions": ["UART_RX", "GPIO"], "description": "UART接收"},
                {"number": 3, "name": "PD1", "pin_type": "bidirectional", "functions": ["UART_TX", "GPIO"], "description": "UART发送"},
                {"number": 4, "name": "PD2", "pin_type": "bidirectional", "functions": ["INT0", "GPIO"], "description": "外部中断0"},
                {"number": 5, "name": "PD3", "pin_type": "bidirectional", "functions": ["INT1", "PWM", "GPIO"], "description": "外部中断1/PWM"},
                {"number": 6, "name": "PD4", "pin_type": "bidirectional", "functions": ["T0", "GPIO"], "description": "定时器0外部输入"},
                {"number": 7, "name": "VCC", "pin_type": "power", "functions": ["电源"], "description": "主电源"},
                {"number": 8, "name": "GND", "pin_type": "power", "functions": ["地"], "description": "地"},
                {"number": 9, "name": "PB6", "pin_type": "bidirectional", "functions": ["XTAL1", "GPIO"], "description": "晶振输入"},
                {"number": 10, "name": "PB7", "pin_type": "bidirectional", "functions": ["XTAL2", "GPIO"], "description": "晶振输出"},
                {"number": 11, "name": "PD5", "pin_type": "bidirectional", "functions": ["PWM", "GPIO"], "description": "PWM输出"},
                {"number": 12, "name": "PD6", "pin_type": "bidirectional", "functions": ["PWM", "GPIO"], "description": "PWM输出"},
                {"number": 13, "name": "PD7", "pin_type": "bidirectional", "functions": ["GPIO"], "description": "通用IO"},
                {"number": 14, "name": "PB0", "pin_type": "bidirectional", "functions": ["ICP1", "GPIO"], "description": "输入捕获"},
                {"number": 15, "name": "PB1", "pin_type": "bidirectional", "functions": ["PWM", "GPIO"], "description": "PWM输出"},
                {"number": 16, "name": "PB2", "pin_type": "bidirectional", "functions": ["SS", "PWM", "GPIO"], "description": "SPI片选"},
                {"number": 17, "name": "PB3", "pin_type": "bidirectional", "functions": ["MOSI", "PWM", "GPIO"], "description": "SPI主出从入"},
                {"number": 18, "name": "PB4", "pin_type": "bidirectional", "functions": ["MISO", "GPIO"], "description": "SPI主入从出"},
                {"number": 19, "name": "PB5", "pin_type": "bidirectional", "functions": ["SCK", "GPIO"], "description": "SPI时钟"},
                {"number": 23, "name": "PC0", "pin_type": "bidirectional", "functions": ["ADC0", "GPIO"], "description": "ADC通道0"},
                {"number": 24, "name": "PC1", "pin_type": "bidirectional", "functions": ["ADC1", "GPIO"], "description": "ADC通道1"},
                {"number": 25, "name": "PC2", "pin_type": "bidirectional", "functions": ["ADC2", "GPIO"], "description": "ADC通道2"},
                {"number": 26, "name": "PC3", "pin_type": "bidirectional", "functions": ["ADC3", "GPIO"], "description": "ADC通道3"},
                {"number": 27, "name": "PC4", "pin_type": "bidirectional", "functions": ["SDA", "ADC4", "GPIO"], "description": "I2C数据/ADC4"},
                {"number": 28, "name": "PC5", "pin_type": "bidirectional", "functions": ["SCL", "ADC5", "GPIO"], "description": "I2C时钟/ADC5"}
            ]
        }
    }

    @classmethod
    def get_pinout(cls, chip_name: str) -> Optional[Dict]:
        """Look up pinout configuration for a chip.

        Lookup order:
          1. Normalize name → exact match in ``STANDARD_PINOUTS``.
          2. Alias lookup in ``PINOUT_ALIASES`` → re-query ``STANDARD_PINOUTS``.
          3. Prefix-based fuzzy match on both dicts.
          4. Return ``None`` if all miss.

        Args:
            chip_name: Chip name (case-insensitive, prefix-tolerant).

        Returns:
            Pinout dict with keys ``pin_count``, ``package``, ``pinout``,
            or ``None`` when no match is found.
        """
        if not chip_name:
            return None

        normalized = cls._normalize_chip_name(chip_name)

        if normalized in cls.STANDARD_PINOUTS:
            return cls.STANDARD_PINOUTS[normalized]

        canonical = cls.PINOUT_ALIASES.get(normalized)
        if canonical and canonical in cls.STANDARD_PINOUTS:
            return cls.STANDARD_PINOUTS[canonical]

        for key in cls.STANDARD_PINOUTS:
            if normalized.startswith(key.rstrip('0123456789')):
                return cls.STANDARD_PINOUTS[key]

        for alias_key, canonical_name in cls.PINOUT_ALIASES.items():
            if normalized.startswith(alias_key.rstrip('0123456789')):
                if canonical_name in cls.STANDARD_PINOUTS:
                    return cls.STANDARD_PINOUTS[canonical_name]

        return None

    @classmethod
    def _normalize_chip_name(cls, chip_name: str) -> str:
        """Normalize a chip name for lookup.

        Steps: uppercase → strip → remove manufacturer prefix →
        remove single-letter package suffix.
        """
        if not chip_name:
            return ""

        name = chip_name.upper().strip()

        manufacturer_prefixes = [
            'SN', 'CD', 'HD', 'UA', 'SE', 'RC', 'TLC',
            'MCP', 'MSP', 'TPS', 'CAT', 'MIC',
        ]
        for prefix in manufacturer_prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        name = cls._SUFFIX_PATTERN.sub('', name)

        return name
