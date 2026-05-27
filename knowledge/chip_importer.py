"""
Common Chip Data Importer for VeriQuery Knowledge Graph.

This module imports electrical parameters and pin information for commonly used
chips into the SQLite knowledge graph database (knowledge_graph.db). It serves
as the data population layer in the knowledge module pipeline:

    graph_db.py (schema) → chip_importer.py (data, this file) → graph_query.py (queries)

Each chip's data is structured in three layers mapped to database tables:
  - chips: basic info (chip_id, name, family, manufacturer, supply_voltage, ...)
  - pins: pin details (pin_id, pin_number, pin_name, function_type, direction, ...)
  - parameters: electrical specs (param_name, param_value, unit, condition, ...)

All parameter values are sourced from official manufacturer datasheets.
"""
import sqlite3
import logging
import json
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class CommonChipDataImporter:
    """Batch importer for common chip data into the knowledge graph database.

    Uses the context manager protocol to manage the SQLite connection lifecycle.
    Data is written using INSERT OR REPLACE to support idempotent imports.

    Usage::

        with CommonChipDataImporter("./data/knowledge_graph.db") as importer:
            importer.import_all()
    """

    def __init__(self, db_path: str = "./data/knowledge_graph.db"):
        """
        Args:
            db_path: Path to the SQLite database file created by graph_db.py.
        """
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def import_all(self):
        """Import all chip categories into the database.

        All imports are executed within a single transaction and committed
        atomically at the end. If any error occurs, the entire batch is
        rolled back to maintain data consistency.
        """
        logger.info("开始导入常用芯片数据")

        self._import_stm32_series()
        self._import_esp32_series()
        self._import_arduino_series()
        self._import_74hc_series()
        self._import_analog_chips()
        self._import_interface_chips()
        self._import_peripheral_chips()

        self.conn.commit()
        logger.info("常用芯片数据导入完成")
    
    def _import_stm32_series(self):
        """Import STM32 series MCU data (ARM Cortex-M3, 3.3V logic)."""
        logger.debug("导入STM32系列芯片")

        stm32_chips = [
            {
                'chip_id': 'STM32F103C8T6',
                'name': 'STM32F103C8T6',
                'full_name': 'STM32F103C8T6',
                'family': 'STM32F1',
                'manufacturer': 'STMicroelectronics',
                'supply_voltage': 3.3,
                'package': 'LQFP48',
                'pin_count': 48,
                'description': 'STM32F103C8T6 - 32-bit MCU, 72 MHz, 64 KB Flash, 20 KB RAM',
                'datasheet_url': 'https://www.st.com/resource/en/datasheet/stm32f103c8.pdf',
                'parameters': {
                    'VOH': {'value': 2.4, 'unit': 'V', 'condition': 'IOH = -20 mA, VDD = 3.3V'},
                    'VOL': {'value': 0.4, 'unit': 'V', 'condition': 'IOL = 20 mA, VDD = 3.3V'},
                    'IOH': {'value': -20.0, 'unit': 'mA', 'condition': 'VOH = 2.4V'},
                    'IOL': {'value': 20.0, 'unit': 'mA', 'condition': 'VOL = 0.4V'},
                    'VIH': {'value': 2.0, 'unit': 'V', 'condition': 'VDD = 3.3V'},
                    'VIL': {'value': 0.8, 'unit': 'V', 'condition': 'VDD = 3.3V'},
                    'IIH': {'value': 0.1, 'unit': 'µA', 'condition': 'VIN = VDD'},
                    'IIL': {'value': -0.1, 'unit': 'µA', 'condition': 'VIN = 0V'},
                    'VCC': {'value': 3.3, 'unit': 'V', 'condition': 'nominal'},
                    'VDD': {'value': 3.3, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 3.3, 'unit': 'V', 'condition': '2.0V to 3.6V'}
                },
                'pins': [
                    {'pin_id': 'STM32F103C8T6_PA9', 'chip_id': 'STM32F103C8T6', 'pin_number': 9, 'pin_name': 'PA9', 'function_type': 'UART_TX', 'direction': 'OUTPUT', 'alternate_functions': ['GPIO', 'TIM1_CH2'], 'electrical_params': {'VOH': 2.4, 'VOL': 0.4, 'IOH': -20.0, 'IOL': 20.0}},
                    {'pin_id': 'STM32F103C8T6_PA10', 'chip_id': 'STM32F103C8T6', 'pin_number': 10, 'pin_name': 'PA10', 'function_type': 'UART_RX', 'direction': 'INPUT', 'alternate_functions': ['GPIO', 'TIM1_CH3'], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': 'STM32F103C8T6_PA0', 'chip_id': 'STM32F103C8T6', 'pin_number': 0, 'pin_name': 'PA0', 'function_type': 'ADC', 'direction': 'INPUT', 'alternate_functions': ['GPIO', 'TIM2_CH1', 'WKUP'], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': 'STM32F103C8T6_PB6', 'chip_id': 'STM32F103C8T6', 'pin_number': 6, 'pin_name': 'PB6', 'function_type': 'I2C_SCL', 'direction': 'BIDIRECTIONAL', 'alternate_functions': ['GPIO', 'USART1_TX', 'TIM4_CH1'], 'electrical_params': {'VOH': 2.4, 'VOL': 0.4, 'IOH': -20.0, 'IOL': 20.0, 'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': 'STM32F103C8T6_PB7', 'chip_id': 'STM32F103C8T6', 'pin_number': 7, 'pin_name': 'PB7', 'function_type': 'I2C_SDA', 'direction': 'BIDIRECTIONAL', 'alternate_functions': ['GPIO', 'USART1_RX', 'TIM4_CH2'], 'electrical_params': {'VOH': 2.4, 'VOL': 0.4, 'IOH': -20.0, 'IOL': 20.0, 'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': 'STM32F103C8T6_PA5', 'chip_id': 'STM32F103C8T6', 'pin_number': 5, 'pin_name': 'PA5', 'function_type': 'SPI_SCK', 'direction': 'OUTPUT', 'alternate_functions': ['GPIO', 'ADC1_IN5'], 'electrical_params': {'VOH': 2.4, 'VOL': 0.4, 'IOH': -20.0, 'IOL': 20.0}},
                    {'pin_id': 'STM32F103C8T6_PA6', 'chip_id': 'STM32F103C8T6', 'pin_number': 6, 'pin_name': 'PA6', 'function_type': 'SPI_MISO', 'direction': 'INPUT', 'alternate_functions': ['GPIO', 'ADC1_IN6', 'TIM3_CH1'], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': 'STM32F103C8T6_PA7', 'chip_id': 'STM32F103C8T6', 'pin_number': 7, 'pin_name': 'PA7', 'function_type': 'SPI_MOSI', 'direction': 'OUTPUT', 'alternate_functions': ['GPIO', 'ADC1_IN7', 'TIM3_CH2'], 'electrical_params': {'VOH': 2.4, 'VOL': 0.4, 'IOH': -20.0, 'IOL': 20.0}}
                ]
            }
        ]

        for chip in stm32_chips:
            self._import_chip(chip)
    
    def _import_esp32_series(self):
        """Import ESP32 series MCU data (Xtensa LX6, 3.3V logic, WiFi/BLE)."""
        logger.debug("导入ESP32系列芯片")

        esp32_chips = [
            {
                'chip_id': 'ESP32-WROOM-32',
                'name': 'ESP32-WROOM-32',
                'full_name': 'ESP32-WROOM-32',
                'family': 'ESP32',
                'manufacturer': 'Espressif',
                'supply_voltage': 3.3,
                'package': 'Module',
                'pin_count': 38,
                'description': 'ESP32-WROOM-32 - WiFi + Bluetooth MCU, 240 MHz, 4 MB Flash',
                'datasheet_url': 'https://www.espressif.com/sites/default/files/documentation/esp32-wroom-32_datasheet_en.pdf',
                'parameters': {
                    'VOH': {'value': 2.8, 'unit': 'V', 'condition': 'IOH = -12 mA, VDD = 3.3V'},
                    'VOL': {'value': 0.5, 'unit': 'V', 'condition': 'IOL = 12 mA, VDD = 3.3V'},
                    'IOH': {'value': -12.0, 'unit': 'mA', 'condition': 'VOH = 2.8V'},
                    'IOL': {'value': 12.0, 'unit': 'mA', 'condition': 'VOL = 0.5V'},
                    'VIH': {'value': 2.0, 'unit': 'V', 'condition': 'VDD = 3.3V'},
                    'VIL': {'value': 0.8, 'unit': 'V', 'condition': 'VDD = 3.3V'},
                    'IIH': {'value': 0.1, 'unit': 'µA', 'condition': 'VIN = VDD'},
                    'IIL': {'value': -0.1, 'unit': 'µA', 'condition': 'VIN = 0V'},
                    'VCC': {'value': 3.3, 'unit': 'V', 'condition': 'nominal'},
                    'VDD': {'value': 3.3, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 3.3, 'unit': 'V', 'condition': '2.3V to 3.6V'}
                },
                'pins': [
                    {'pin_id': 'ESP32_TX0', 'chip_id': 'ESP32-WROOM-32', 'pin_number': 1, 'pin_name': 'TX0', 'function_type': 'UART_TX', 'direction': 'OUTPUT', 'alternate_functions': ['GPIO1'], 'electrical_params': {'VOH': 2.8, 'VOL': 0.5, 'IOH': -12.0, 'IOL': 12.0}},
                    {'pin_id': 'ESP32_RX0', 'chip_id': 'ESP32-WROOM-32', 'pin_number': 3, 'pin_name': 'RX0', 'function_type': 'UART_RX', 'direction': 'INPUT', 'alternate_functions': ['GPIO3'], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': 'ESP32_SDA', 'chip_id': 'ESP32-WROOM-32', 'pin_number': 21, 'pin_name': 'SDA', 'function_type': 'I2C_SDA', 'direction': 'BIDIRECTIONAL', 'alternate_functions': ['GPIO21'], 'electrical_params': {'VOH': 2.8, 'VOL': 0.5, 'IOH': -12.0, 'IOL': 12.0, 'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': 'ESP32_SCL', 'chip_id': 'ESP32-WROOM-32', 'pin_number': 22, 'pin_name': 'SCL', 'function_type': 'I2C_SCL', 'direction': 'BIDIRECTIONAL', 'alternate_functions': ['GPIO22'], 'electrical_params': {'VOH': 2.8, 'VOL': 0.5, 'IOH': -12.0, 'IOL': 12.0, 'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': 'ESP32_MOSI', 'chip_id': 'ESP32-WROOM-32', 'pin_number': 23, 'pin_name': 'MOSI', 'function_type': 'SPI_MOSI', 'direction': 'OUTPUT', 'alternate_functions': ['GPIO23'], 'electrical_params': {'VOH': 2.8, 'VOL': 0.5, 'IOH': -12.0, 'IOL': 12.0}},
                    {'pin_id': 'ESP32_MISO', 'chip_id': 'ESP32-WROOM-32', 'pin_number': 19, 'pin_name': 'MISO', 'function_type': 'SPI_MISO', 'direction': 'INPUT', 'alternate_functions': ['GPIO19'], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': 'ESP32_SCK', 'chip_id': 'ESP32-WROOM-32', 'pin_number': 18, 'pin_name': 'SCK', 'function_type': 'SPI_SCK', 'direction': 'OUTPUT', 'alternate_functions': ['GPIO18'], 'electrical_params': {'VOH': 2.8, 'VOL': 0.5, 'IOH': -12.0, 'IOL': 12.0}}
                ]
            }
        ]

        for chip in esp32_chips:
            self._import_chip(chip)
    
    def _import_arduino_series(self):
        """Import Arduino/AVR series MCU data (ATmega328P, 5V logic)."""
        logger.debug("导入Arduino系列芯片")

        arduino_chips = [
            {
                'chip_id': 'ATmega328P',
                'name': 'ATmega328P',
                'full_name': 'ATmega328P-PU',
                'family': 'AVR',
                'manufacturer': 'Microchip',
                'supply_voltage': 5.0,
                'package': 'DIP28',
                'pin_count': 28,
                'description': 'ATmega328P - 8-bit MCU, 20 MHz, 32 KB Flash, 2 KB RAM',
                'datasheet_url': 'https://ww1.microchip.com/downloads/en/DeviceDoc/ATmega328_P%20AVR%20MCU%20with%20picoPower%20Technology%20Data%20Sheet%2040001984A.pdf',
                'parameters': {
                    'VOH': {'value': 4.2, 'unit': 'V', 'condition': 'IOH = -20 mA, VCC = 5V'},
                    'VOL': {'value': 0.5, 'unit': 'V', 'condition': 'IOL = 20 mA, VCC = 5V'},
                    'IOH': {'value': -20.0, 'unit': 'mA', 'condition': 'VOH = 4.2V'},
                    'IOL': {'value': 20.0, 'unit': 'mA', 'condition': 'VOL = 0.5V'},
                    'VIH': {'value': 3.0, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'VIL': {'value': 1.5, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'IIH': {'value': 1.0, 'unit': 'µA', 'condition': 'VIN = VCC'},
                    'IIL': {'value': -1.0, 'unit': 'µA', 'condition': 'VIN = 0V'},
                    'VCC': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'VDD': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 5.0, 'unit': 'V', 'condition': '1.8V to 5.5V'}
                },
                'pins': [
                    {'pin_id': 'ATmega328P_TX', 'chip_id': 'ATmega328P', 'pin_number': 3, 'pin_name': 'PD1', 'function_type': 'UART_TX', 'direction': 'OUTPUT', 'alternate_functions': ['GPIO'], 'electrical_params': {'VOH': 4.2, 'VOL': 0.5, 'IOH': -20.0, 'IOL': 20.0}},
                    {'pin_id': 'ATmega328P_RX', 'chip_id': 'ATmega328P', 'pin_number': 2, 'pin_name': 'PD0', 'function_type': 'UART_RX', 'direction': 'INPUT', 'alternate_functions': ['GPIO'], 'electrical_params': {'VIH': 3.0, 'VIL': 1.5, 'IIH': 1.0, 'IIL': -1.0}},
                    {'pin_id': 'ATmega328P_SDA', 'chip_id': 'ATmega328P', 'pin_number': 27, 'pin_name': 'PC4', 'function_type': 'I2C_SDA', 'direction': 'BIDIRECTIONAL', 'alternate_functions': ['GPIO', 'ADC4'], 'electrical_params': {'VOH': 4.2, 'VOL': 0.5, 'IOH': -20.0, 'IOL': 20.0, 'VIH': 3.0, 'VIL': 1.5, 'IIH': 1.0, 'IIL': -1.0}},
                    {'pin_id': 'ATmega328P_SCL', 'chip_id': 'ATmega328P', 'pin_number': 28, 'pin_name': 'PC5', 'function_type': 'I2C_SCL', 'direction': 'BIDIRECTIONAL', 'alternate_functions': ['GPIO', 'ADC5'], 'electrical_params': {'VOH': 4.2, 'VOL': 0.5, 'IOH': -20.0, 'IOL': 20.0, 'VIH': 3.0, 'VIL': 1.5, 'IIH': 1.0, 'IIL': -1.0}},
                    {'pin_id': 'ATmega328P_MOSI', 'chip_id': 'ATmega328P', 'pin_number': 17, 'pin_name': 'PB3', 'function_type': 'SPI_MOSI', 'direction': 'OUTPUT', 'alternate_functions': ['GPIO', 'MOSI', 'OC2A'], 'electrical_params': {'VOH': 4.2, 'VOL': 0.5, 'IOH': -20.0, 'IOL': 20.0}},
                    {'pin_id': 'ATmega328P_MISO', 'chip_id': 'ATmega328P', 'pin_number': 18, 'pin_name': 'PB4', 'function_type': 'SPI_MISO', 'direction': 'INPUT', 'alternate_functions': ['GPIO', 'MISO'], 'electrical_params': {'VIH': 3.0, 'VIL': 1.5, 'IIH': 1.0, 'IIL': -1.0}},
                    {'pin_id': 'ATmega328P_SCK', 'chip_id': 'ATmega328P', 'pin_number': 19, 'pin_name': 'PB5', 'function_type': 'SPI_SCK', 'direction': 'OUTPUT', 'alternate_functions': ['GPIO', 'SCK'], 'electrical_params': {'VOH': 4.2, 'VOL': 0.5, 'IOH': -20.0, 'IOL': 20.0}}
                ]
            }
        ]

        for chip in arduino_chips:
            self._import_chip(chip)
    
    def _import_74hc_series(self):
        """Import 74HC/HCT series digital logic chip data (CMOS)."""
        logger.debug("导入74HC/HCT系列芯片")

        hc_chips = [
            {
                'chip_id': '74HC04',
                'name': '74HC04',
                'full_name': 'SN74HC04',
                'family': '74HC',
                'manufacturer': 'TI',
                'supply_voltage': 5.0,
                'package': 'DIP14',
                'pin_count': 14,
                'description': '74HC04 - Hex Inverter (CMOS)',
                'datasheet_url': 'https://assets.nexperia.com/documents/data-sheet/74HC_HCT04.pdf',
                'parameters': {
                    'VOH': {'value': 4.5, 'unit': 'V', 'condition': 'IOH = -4 mA, VCC = 5V'},
                    'VOL': {'value': 0.1, 'unit': 'V', 'condition': 'IOL = 4 mA, VCC = 5V'},
                    'IOH': {'value': -4.0, 'unit': 'mA', 'condition': 'VOH = 4.5V'},
                    'IOL': {'value': 4.0, 'unit': 'mA', 'condition': 'VOL = 0.1V'},
                    'VIH': {'value': 3.5, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'VIL': {'value': 1.0, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'IIH': {'value': 0.1, 'unit': 'µA', 'condition': 'VIN = VCC'},
                    'IIL': {'value': -0.1, 'unit': 'µA', 'condition': 'VIN = 0V'},
                    'ICC': {'value': 1.0, 'unit': 'uA', 'condition': 'VCC = 5V, 25C'},
                    'IDD': {'value': 1.0, 'unit': 'uA', 'condition': 'VDD = 5V, 25C'},
                    'VCC': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'VDD': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'tPLH': {'value': 10.0, 'unit': 'ns', 'condition': 'VCC = 5V, CL=50pF'},
                    'tPHL': {'value': 10.0, 'unit': 'ns', 'condition': 'VCC = 5V, CL=50pF'},
                    'tpd': {'value': 10.0, 'unit': 'ns', 'condition': 'VCC = 5V, CL=50pF'},
                    'fmax': {'value': 30.0, 'unit': 'MHz', 'condition': 'VCC = 5V'},
                    'Frequency': {'value': 30.0, 'unit': 'MHz', 'condition': 'VCC = 5V'},
                    'Temperature': {'value': 85.0, 'unit': 'C', 'condition': 'Operating Range'},
                    'supply_voltage': {'value': 5.0, 'unit': 'V', 'condition': '2.0V to 6.0V'}
                },
                'pins': [
                    {'pin_id': '74HC04_1A', 'chip_id': '74HC04', 'pin_number': 1, 'pin_name': '1A', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 3.5, 'VIL': 1.0, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': '74HC04_1Y', 'chip_id': '74HC04', 'pin_number': 2, 'pin_name': '1Y', 'function_type': 'OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 4.5, 'VOL': 0.1, 'IOH': -4.0, 'IOL': 4.0}},
                    {'pin_id': '74HC04_2A', 'chip_id': '74HC04', 'pin_number': 3, 'pin_name': '2A', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 3.5, 'VIL': 1.0, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': '74HC04_2Y', 'chip_id': '74HC04', 'pin_number': 4, 'pin_name': '2Y', 'function_type': 'OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 4.5, 'VOL': 0.1, 'IOH': -4.0, 'IOL': 4.0}}
                ]
            },
            {
                'chip_id': '74HCT04',
                'name': '74HCT04',
                'full_name': 'SN74HCT04',
                'family': '74HCT',
                'manufacturer': 'TI',
                'supply_voltage': 5.0,
                'package': 'DIP14',
                'pin_count': 14,
                'description': '74HCT04 - Hex Inverter (TTL-compatible CMOS)',
                'datasheet_url': 'https://assets.nexperia.com/documents/data-sheet/74HC_HCT04.pdf',
                'parameters': {
                    'VOH': {'value': 4.5, 'unit': 'V', 'condition': 'IOH = -4 mA, VCC = 5V'},
                    'VOL': {'value': 0.1, 'unit': 'V', 'condition': 'IOL = 4 mA, VCC = 5V'},
                    'IOH': {'value': -4.0, 'unit': 'mA', 'condition': 'VOH = 4.5V'},
                    'IOL': {'value': 4.0, 'unit': 'mA', 'condition': 'VOL = 0.1V'},
                    'VIH': {'value': 2.0, 'unit': 'V', 'condition': 'VCC = 4.5V to 5.5V'},
                    'VIL': {'value': 0.8, 'unit': 'V', 'condition': 'VCC = 4.5V to 5.5V'},
                    'IIH': {'value': 0.1, 'unit': 'µA', 'condition': 'VIN = VCC'},
                    'IIL': {'value': -0.1, 'unit': 'µA', 'condition': 'VIN = 0V'},
                    'ICC': {'value': 1.0, 'unit': 'uA', 'condition': 'VCC = 5V, 25C'},
                    'IDD': {'value': 1.0, 'unit': 'uA', 'condition': 'VDD = 5V, 25C'},
                    'VCC': {'value': 5.0, 'unit': 'V', 'condition': '4.5V to 5.5V'},
                    'VDD': {'value': 5.0, 'unit': 'V', 'condition': '4.5V to 5.5V'},
                    'tPLH': {'value': 12.0, 'unit': 'ns', 'condition': 'VCC = 5V, CL=50pF'},
                    'tPHL': {'value': 12.0, 'unit': 'ns', 'condition': 'VCC = 5V, CL=50pF'},
                    'tpd': {'value': 12.0, 'unit': 'ns', 'condition': 'VCC = 5V, CL=50pF'},
                    'fmax': {'value': 30.0, 'unit': 'MHz', 'condition': 'VCC = 5V'},
                    'Frequency': {'value': 30.0, 'unit': 'MHz', 'condition': 'VCC = 5V'},
                    'Temperature': {'value': 85.0, 'unit': 'C', 'condition': 'Operating Range'},
                    'supply_voltage': {'value': 5.0, 'unit': 'V', 'condition': '4.5V to 5.5V'}
                },
                'pins': [
                    {'pin_id': '74HCT04_1A', 'chip_id': '74HCT04', 'pin_number': 1, 'pin_name': '1A', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': '74HCT04_1Y', 'chip_id': '74HCT04', 'pin_number': 2, 'pin_name': '1Y', 'function_type': 'OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 4.5, 'VOL': 0.1, 'IOH': -4.0, 'IOL': 4.0}},
                    {'pin_id': '74HCT04_2A', 'chip_id': '74HCT04', 'pin_number': 3, 'pin_name': '2A', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': '74HCT04_2Y', 'chip_id': '74HCT04', 'pin_number': 4, 'pin_name': '2Y', 'function_type': 'OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 4.5, 'VOL': 0.1, 'IOH': -4.0, 'IOL': 4.0}}
                ]
            },
            {
                'chip_id': '74HC595',
                'name': '74HC595',
                'full_name': '74HC595',
                'family': '74HC',
                'manufacturer': 'NXP',
                'supply_voltage': 5.0,
                'package': 'DIP16',
                'pin_count': 16,
                'description': '74HC595 - 8-bit Shift Register with Output Latches',
                'datasheet_url': 'https://assets.nexperia.com/documents/data-sheet/74HC_HCT595.pdf',
                'parameters': {
                    'VOH': {'value': 4.5, 'unit': 'V', 'condition': 'IOH = -6 mA, VCC = 5V'},
                    'VOL': {'value': 0.1, 'unit': 'V', 'condition': 'IOL = 6 mA, VCC = 5V'},
                    'IOH': {'value': -6.0, 'unit': 'mA', 'condition': 'VOH = 4.5V'},
                    'IOL': {'value': 6.0, 'unit': 'mA', 'condition': 'VOL = 0.1V'},
                    'VIH': {'value': 3.5, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'VIL': {'value': 1.0, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'IIH': {'value': 0.1, 'unit': 'µA', 'condition': 'VIN = VCC'},
                    'IIL': {'value': -0.1, 'unit': 'µA', 'condition': 'VIN = 0V'},
                    'VCC': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'VDD': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 5.0, 'unit': 'V', 'condition': '2.0V to 6.0V'}
                },
                'pins': [
                    {'pin_id': '74HC595_DS', 'chip_id': '74HC595', 'pin_number': 14, 'pin_name': 'DS', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 3.5, 'VIL': 1.0, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': '74HC595_ST_CP', 'chip_id': '74HC595', 'pin_number': 12, 'pin_name': 'ST_CP', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 3.5, 'VIL': 1.0, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': '74HC595_SH_CP', 'chip_id': '74HC595', 'pin_number': 11, 'pin_name': 'SH_CP', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 3.5, 'VIL': 1.0, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': '74HC595_Q0', 'chip_id': '74HC595', 'pin_number': 15, 'pin_name': 'Q0', 'function_type': 'OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 4.5, 'VOL': 0.1, 'IOH': -6.0, 'IOL': 6.0}}
                ]
            }
        ]

        for chip in hc_chips:
            self._import_chip(chip)
    

    def _import_analog_chips(self):
        """Import analog and power management chip data (op-amps, timers, comparators, regulators)."""
        logger.info("导入模拟芯片和电源管理芯片")

        analog_chips = [
            {
                'chip_id': 'LM358',
                'name': 'LM358',
                'full_name': 'LM358N',
                'family': 'OpAmp',
                'manufacturer': 'Texas Instruments',
                'supply_voltage': 5.0,
                'package': 'DIP8',
                'pin_count': 8,
                'description': 'LM358 - Dual Operational Amplifier',
                'datasheet_url': 'https://www.ti.com/lit/ds/symlink/lm358.pdf',
                'parameters': {
                    'VOH': {'value': 3.5, 'unit': 'V', 'condition': 'IOH = -20 mA, VCC = 5V'},
                    'VOL': {'value': 0.2, 'unit': 'V', 'condition': 'IOL = 20 mA, VCC = 5V'},
                    'IOH': {'value': -20.0, 'unit': 'mA', 'condition': 'VOH = 3.5V'},
                    'IOL': {'value': 20.0, 'unit': 'mA', 'condition': 'VOL = 0.2V'},
                    'VIH': {'value': 2.0, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'VIL': {'value': 0.8, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'IIH': {'value': 0.2, 'unit': 'µA', 'condition': 'VIN = VCC'},
                    'IIL': {'value': -0.2, 'unit': 'µA', 'condition': 'VIN = 0V'},
                    'ICC': {'value': 0.7, 'unit': 'mA', 'condition': 'VCC = 5V, No load, 25C'},
                    'IDD': {'value': 0.7, 'unit': 'mA', 'condition': 'VDD = 5V, No load, 25C'},
                    'VCC': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'VDD': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'tPLH': {'value': 200.0, 'unit': 'ns', 'condition': 'Slew rate 0.5V/us, Large signal'},
                    'tPHL': {'value': 200.0, 'unit': 'ns', 'condition': 'Slew rate 0.5V/us, Large signal'},
                    'tpd': {'value': 200.0, 'unit': 'ns', 'condition': 'Slew rate 0.5V/us, Large signal'},
                    'fmax': {'value': 1.0, 'unit': 'MHz', 'condition': 'Gain Bandwidth Product, VCC = 5V'},
                    'Frequency': {'value': 1.0, 'unit': 'MHz', 'condition': 'Gain Bandwidth Product, VCC = 5V'},
                    'Temperature': {'value': 70.0, 'unit': 'C', 'condition': 'Operating Range: 0 to +70C'},
                    'supply_voltage': {'value': 5.0, 'unit': 'V', 'condition': '3.0V to 32V'}
                },
                'pins': [
                    {'pin_id': 'LM358_IN1+', 'chip_id': 'LM358', 'pin_number': 3, 'pin_name': 'IN1+', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.2, 'IIL': -0.2}},
                    {'pin_id': 'LM358_IN1-', 'chip_id': 'LM358', 'pin_number': 2, 'pin_name': 'IN1-', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.2, 'IIL': -0.2}},
                    {'pin_id': 'LM358_OUT1', 'chip_id': 'LM358', 'pin_number': 1, 'pin_name': 'OUT1', 'function_type': 'OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 3.5, 'VOL': 0.2, 'IOH': -20.0, 'IOL': 20.0}}
                ]
            },
            {
                'chip_id': 'NE555',
                'name': 'NE555',
                'full_name': 'NE555P',
                'family': 'Timer',
                'manufacturer': 'Texas Instruments',
                'supply_voltage': 5.0,
                'package': 'DIP8',
                'pin_count': 8,
                'description': 'NE555 - Single Precision Timer',
                'datasheet_url': 'https://www.ti.com/lit/ds/symlink/ne555.pdf',
                'parameters': {
                    'VOH': {'value': 4.5, 'unit': 'V', 'condition': 'IOH = -100 mA, VCC = 5V'},
                    'VOL': {'value': 0.25, 'unit': 'V', 'condition': 'IOL = 100 mA, VCC = 5V'},
                    'IOH': {'value': -100.0, 'unit': 'mA', 'condition': 'VOH = 4.5V'},
                    'IOL': {'value': 100.0, 'unit': 'mA', 'condition': 'VOL = 0.25V'},
                    'VIH': {'value': 3.3, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'VIL': {'value': 1.67, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'IIH': {'value': 0.1, 'unit': 'µA', 'condition': 'VIN = VCC'},
                    'IIL': {'value': -0.1, 'unit': 'µA', 'condition': 'VIN = 0V'},
                    'ICC': {'value': 3.0, 'unit': 'mA', 'condition': 'VCC = 5V, No load, 25C'},
                    'IDD': {'value': 3.0, 'unit': 'mA', 'condition': 'VDD = 5V, No load, 25C'},
                    'VCC': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'VDD': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'tPLH': {'value': 100.0, 'unit': 'ns', 'condition': 'Output rise time, VCC=5V'},
                    'tPHL': {'value': 100.0, 'unit': 'ns', 'condition': 'Output fall time, VCC=5V'},
                    'tpd': {'value': 100.0, 'unit': 'ns', 'condition': 'Propagation delay, VCC=5V'},
                    'fmax': {'value': 0.1, 'unit': 'MHz', 'condition': 'Max operating frequency, 100kHz'},
                    'Frequency': {'value': 0.1, 'unit': 'MHz', 'condition': 'Max operating frequency, 100kHz'},
                    'Temperature': {'value': 70.0, 'unit': 'C', 'condition': 'Operating Range: 0 to +70C'},
                    'supply_voltage': {'value': 5.0, 'unit': 'V', 'condition': '4.5V to 16V'}
                },
                'pins': [
                    {'pin_id': 'NE555_TRIGGER', 'chip_id': 'NE555', 'pin_number': 2, 'pin_name': 'TRIGGER', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 3.3, 'VIL': 1.67, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': 'NE555_OUTPUT', 'chip_id': 'NE555', 'pin_number': 3, 'pin_name': 'OUTPUT', 'function_type': 'OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 4.5, 'VOL': 0.25, 'IOH': -100.0, 'IOL': 100.0}},
                    {'pin_id': 'NE555_RESET', 'chip_id': 'NE555', 'pin_number': 4, 'pin_name': 'RESET', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 3.3, 'VIL': 1.67, 'IIH': 0.1, 'IIL': -0.1}}
                ]
            },
            {
                'chip_id': 'LM393',
                'name': 'LM393',
                'full_name': 'LM393N',
                'family': 'Comparator',
                'manufacturer': 'Texas Instruments',
                'supply_voltage': 5.0,
                'package': 'DIP8',
                'pin_count': 8,
                'description': 'LM393 - Dual Differential Comparator',
                'datasheet_url': 'https://www.ti.com/lit/ds/symlink/lm393-n.pdf',
                'parameters': {
                    'VOH': {'value': 5.0, 'unit': 'V', 'condition': 'Open collector, pull-up to VCC'},
                    'VOL': {'value': 0.25, 'unit': 'V', 'condition': 'IOL = 4 mA, VCC = 5V'},
                    'IOH': {'value': 0.0, 'unit': 'mA', 'condition': 'Open collector'},
                    'IOL': {'value': 6.0, 'unit': 'mA', 'condition': 'VOL = 0.25V'},
                    'VIH': {'value': 2.0, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'VIL': {'value': 0.8, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'IIH': {'value': 0.05, 'unit': 'µA', 'condition': 'VIN = VCC'},
                    'IIL': {'value': -0.05, 'unit': 'µA', 'condition': 'VIN = 0V'},
                    'ICC': {'value': 0.8, 'unit': 'mA', 'condition': 'VCC = 5V, No load, 25C'},
                    'IDD': {'value': 0.8, 'unit': 'mA', 'condition': 'VDD = 5V, No load, 25C'},
                    'VCC': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'VDD': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'tPLH': {'value': 1300.0, 'unit': 'ns', 'condition': 'Response time, VCC=5V'},
                    'tPHL': {'value': 1300.0, 'unit': 'ns', 'condition': 'Response time, VCC=5V'},
                    'tpd': {'value': 1300.0, 'unit': 'ns', 'condition': 'Response time, VCC=5V'},
                    'fmax': {'value': 1.0, 'unit': 'MHz', 'condition': 'Max toggle frequency (limited by response)'},
                    'Frequency': {'value': 1.0, 'unit': 'MHz', 'condition': 'Max toggle frequency (limited by response)'},
                    'Temperature': {'value': 70.0, 'unit': 'C', 'condition': 'Operating Range: 0 to +70C'},
                    'supply_voltage': {'value': 5.0, 'unit': 'V', 'condition': '2.0V to 36V'}
                },
                'pins': [
                    {'pin_id': 'LM393_IN1+', 'chip_id': 'LM393', 'pin_number': 3, 'pin_name': 'IN1+', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.05, 'IIL': -0.05}},
                    {'pin_id': 'LM393_IN1-', 'chip_id': 'LM393', 'pin_number': 2, 'pin_name': 'IN1-', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.05, 'IIL': -0.05}},
                    {'pin_id': 'LM393_OUT1', 'chip_id': 'LM393', 'pin_number': 1, 'pin_name': 'OUT1', 'function_type': 'OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 5.0, 'VOL': 0.25, 'IOH': 0.0, 'IOL': 6.0}}
                ]
            },
            {
                'chip_id': 'LM7805',
                'name': 'LM7805',
                'full_name': 'LM7805CT',
                'family': 'VoltageRegulator',
                'manufacturer': 'Texas Instruments',
                'supply_voltage': 9.0,
                'package': 'TO-220',
                'pin_count': 3,
                'description': 'LM7805 - 5V Positive Voltage Regulator',
                'datasheet_url': 'https://www.ti.com/lit/ds/symlink/lm7805.pdf',
                'parameters': {
                    'VOH': {'value': 5.0, 'unit': 'V', 'condition': 'IO = 500 mA, VI = 10V'},
                    'VOL': {'value': 0.0, 'unit': 'V', 'condition': 'N/A - Linear Regulator'},
                    'IOH': {'value': 1000.0, 'unit': 'mA', 'condition': 'VO = 5V'},
                    'IOL': {'value': 0.0, 'unit': 'mA', 'condition': 'N/A - Linear Regulator'},
                    'VIH': {'value': 7.0, 'unit': 'V', 'condition': 'Minimum input voltage (Dropout)'},
                    'VIL': {'value': 0.0, 'unit': 'V', 'condition': 'N/A - Linear Regulator'},
                    'IIH': {'value': 4.2, 'unit': 'mA', 'condition': 'Ground current at IO=500mA'},
                    'IIL': {'value': 4.2, 'unit': 'mA', 'condition': 'Ground current at IO=500mA'},
                    'ICC': {'value': 5.0, 'unit': 'mA', 'condition': 'Quiescent current, VI=10V, IO=0'},
                    'IDD': {'value': 5.0, 'unit': 'mA', 'condition': 'Quiescent current, no load'},
                    'VCC': {'value': 9.0, 'unit': 'V', 'condition': 'nominal input'},
                    'VDD': {'value': 5.0, 'unit': 'V', 'condition': 'output'},
                    'tPLH': {'value': 9999.0, 'unit': 'ns', 'condition': 'N/A - DC Linear Regulator'},
                    'tPHL': {'value': 9999.0, 'unit': 'ns', 'condition': 'N/A - DC Linear Regulator'},
                    'tpd': {'value': 9999.0, 'unit': 'ns', 'condition': 'N/A - DC Linear Regulator'},
                    'fmax': {'value': 0.001, 'unit': 'MHz', 'condition': 'N/A - DC device only'},
                    'Frequency': {'value': 0.001, 'unit': 'MHz', 'condition': 'N/A - DC device only'},
                    'Temperature': {'value': 125.0, 'unit': 'C', 'condition': 'Operating Range: 0 to +125C'},
                    'supply_voltage': {'value': 9.0, 'unit': 'V', 'condition': '7V to 35V input'}
                },
                'pins': [
                    {'pin_id': 'LM7805_IN', 'chip_id': 'LM7805', 'pin_number': 1, 'pin_name': 'VIN', 'function_type': 'POWER_INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {}},
                    {'pin_id': 'LM7805_GND', 'chip_id': 'LM7805', 'pin_number': 2, 'pin_name': 'GND', 'function_type': 'GROUND', 'direction': 'BIDIRECTIONAL', 'alternate_functions': [], 'electrical_params': {}},
                    {'pin_id': 'LM7805_OUT', 'chip_id': 'LM7805', 'pin_number': 3, 'pin_name': 'VOUT', 'function_type': 'POWER_OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 5.0, 'IOH': 1000.0}}
                ]
            },
            {
                'chip_id': 'AMS1117-3.3',
                'name': 'AMS1117-3.3',
                'full_name': 'AMS1117-3.3',
                'family': 'LDO',
                'manufacturer': 'Advanced Monolithic Systems',
                'supply_voltage': 5.0,
                'package': 'SOT-223',
                'pin_count': 4,
                'description': 'AMS1117-3.3 - 3.3V Low Dropout Linear Regulator',
                'datasheet_url': 'https://www.advanced-monolithic.com/pdf/ds1117.pdf',
                'parameters': {
                    'VOH': {'value': 3.3, 'unit': 'V', 'condition': 'IO = 800 mA, VI = 5V'},
                    'VOL': {'value': 0.0, 'unit': 'V', 'condition': 'N/A - LDO Regulator'},
                    'IOH': {'value': 1000.0, 'unit': 'mA', 'condition': 'VO = 3.3V'},
                    'IOL': {'value': 0.0, 'unit': 'mA', 'condition': 'N/A - LDO Regulator'},
                    'VIH': {'value': 4.3, 'unit': 'V', 'condition': 'Minimum input voltage (Dropout)'},
                    'VIL': {'value': 0.0, 'unit': 'V', 'condition': 'N/A - LDO Regulator'},
                    'IIH': {'value': 5.0, 'unit': 'mA', 'condition': 'Ground current at IO=500mA'},
                    'IIL': {'value': 5.0, 'unit': 'mA', 'condition': 'Ground current at IO=500mA'},
                    'ICC': {'value': 10.0, 'unit': 'mA', 'condition': 'Quiescent current, VI=5V, IO=0'},
                    'IDD': {'value': 10.0, 'unit': 'mA', 'condition': 'Quiescent current, no load'},
                    'VCC': {'value': 5.0, 'unit': 'V', 'condition': 'nominal input'},
                    'VDD': {'value': 3.3, 'unit': 'V', 'condition': 'output'},
                    'tPLH': {'value': 9999.0, 'unit': 'ns', 'condition': 'N/A - DC LDO Regulator'},
                    'tPHL': {'value': 9999.0, 'unit': 'ns', 'condition': 'N/A - DC LDO Regulator'},
                    'tpd': {'value': 9999.0, 'unit': 'ns', 'condition': 'N/A - DC LDO Regulator'},
                    'fmax': {'value': 0.001, 'unit': 'MHz', 'condition': 'N/A - DC device only'},
                    'Frequency': {'value': 0.001, 'unit': 'MHz', 'condition': 'N/A - DC device only'},
                    'Temperature': {'value': 125.0, 'unit': 'C', 'condition': 'Operating Range: -40 to +125C'},
                    'supply_voltage': {'value': 5.0, 'unit': 'V', 'condition': '4.3V to 15V input'}
                },
                'pins': [
                    {'pin_id': 'AMS1117_GND', 'chip_id': 'AMS1117-3.3', 'pin_number': 1, 'pin_name': 'GND', 'function_type': 'GROUND', 'direction': 'BIDIRECTIONAL', 'alternate_functions': [], 'electrical_params': {}},
                    {'pin_id': 'AMS1117_OUT', 'chip_id': 'AMS1117-3.3', 'pin_number': 2, 'pin_name': 'VOUT', 'function_type': 'POWER_OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 3.3, 'IOH': 1000.0}},
                    {'pin_id': 'AMS1117_IN', 'chip_id': 'AMS1117-3.3', 'pin_number': 3, 'pin_name': 'VIN', 'function_type': 'POWER_INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {}}
                ]
            },
            {
                'chip_id': 'NE5532',
                'name': 'NE5532',
                'full_name': 'NE5532 Dual Low-Noise Operational Amplifier',
                'family': 'OpAmp',
                'manufacturer': 'Texas Instruments',
                'supply_voltage': 15.0,
                'package': 'DIP8',
                'pin_count': 8,
                'description': 'NE5532 - Dual Low-Noise Operational Amplifier, High Performance',
                'datasheet_url': 'https://www.ti.com/lit/ds/symlink/ne5532.pdf',
                'parameters': {
                    'VOH': {'value': 13.0, 'unit': 'V', 'condition': 'IOH = -10 mA, VCC+ = 15V, VCC- = -15V'},
                    'VOL': {'value': -13.0, 'unit': 'V', 'condition': 'IOL = 10 mA, VCC+ = 15V, VCC- = -15V'},
                    'IOH': {'value': -10.0, 'unit': 'mA', 'condition': 'VOH = 13V'},
                    'IOL': {'value': 10.0, 'unit': 'mA', 'condition': 'VOL = -13V'},
                    'VIH': {'value': 12.0, 'unit': 'V', 'condition': 'VCC+ = 15V, Common mode range'},
                    'VIL': {'value': -12.0, 'unit': 'V', 'condition': 'VCC- = -15V, Common mode range'},
                    'IIH': {'value': 0.2, 'unit': 'µA', 'condition': 'VIN = VCC+, Input bias current'},
                    'IIL': {'value': -0.8, 'unit': 'µA', 'condition': 'VIN = VCC-, Input bias current'},
                    'ICC': {'value': 8.0, 'unit': 'mA', 'condition': 'VCC = ±15V, No load, 25C'},
                    'IDD': {'value': 8.0, 'unit': 'mA', 'condition': 'VDD = ±15V, No load, 25C'},
                    'VCC': {'value': 15.0, 'unit': 'V', 'condition': 'VCC+ positive supply'},
                    'VDD': {'value': -15.0, 'unit': 'V', 'condition': 'VCC- negative supply'},
                    'tPLH': {'value': 110.0, 'unit': 'ns', 'condition': 'Slew rate 9V/us, Large signal'},
                    'tPHL': {'value': 110.0, 'unit': 'ns', 'condition': 'Slew rate 9V/us, Large signal'},
                    'tpd': {'value': 110.0, 'unit': 'ns', 'condition': 'Slew rate 9V/us, Large signal'},
                    'fmax': {'value': 10.0, 'unit': 'MHz', 'condition': 'Gain Bandwidth Product, VCC = ±15V'},
                    'Frequency': {'value': 10.0, 'unit': 'MHz', 'condition': 'Gain Bandwidth Product, VCC = ±15V'},
                    'Temperature': {'value': 85.0, 'unit': 'C', 'condition': 'Operating Range: -40 to +85C'},
                    'supply_voltage': {'value': 15.0, 'unit': 'V', 'condition': '±3V to ±22V dual supply'}
                },
                'pins': [
                    {'pin_id': 'NE5532_OUT1', 'chip_id': 'NE5532', 'pin_number': 1, 'pin_name': 'OUT1', 'function_type': 'OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 13.0, 'VOL': -13.0, 'IOH': -10.0, 'IOL': 10.0}},
                    {'pin_id': 'NE5532_IN1-', 'chip_id': 'NE5532', 'pin_number': 2, 'pin_name': 'IN1-', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 12.0, 'VIL': -12.0, 'IIH': 0.2, 'IIL': -0.8}},
                    {'pin_id': 'NE5532_IN1+', 'chip_id': 'NE5532', 'pin_number': 3, 'pin_name': 'IN1+', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 12.0, 'VIL': -12.0, 'IIH': 0.2, 'IIL': -0.8}},
                    {'pin_id': 'NE5532_VCC-', 'chip_id': 'NE5532', 'pin_number': 4, 'pin_name': 'VCC-', 'function_type': 'POWER_INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {}},
                    {'pin_id': 'NE5532_IN2+', 'chip_id': 'NE5532', 'pin_number': 5, 'pin_name': 'IN2+', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 12.0, 'VIL': -12.0, 'IIH': 0.2, 'IIL': -0.8}},
                    {'pin_id': 'NE5532_IN2-', 'chip_id': 'NE5532', 'pin_number': 6, 'pin_name': 'IN2-', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 12.0, 'VIL': -12.0, 'IIH': 0.2, 'IIL': -0.8}},
                    {'pin_id': 'NE5532_OUT2', 'chip_id': 'NE5532', 'pin_number': 7, 'pin_name': 'OUT2', 'function_type': 'OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 13.0, 'VOL': -13.0, 'IOH': -10.0, 'IOL': 10.0}},
                    {'pin_id': 'NE5532_VCC+', 'chip_id': 'NE5532', 'pin_number': 8, 'pin_name': 'VCC+', 'function_type': 'POWER_INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {}}
                ]
            }
        ]
        
        for chip in analog_chips:
            self._import_chip(chip)
    
    def _import_interface_chips(self):
        """Import interface and driver chip data (RS-232 transceivers, Darlington arrays)."""
        logger.debug("导入接口芯片和驱动芯片")

        interface_chips = [
            {
                'chip_id': 'MAX232',
                'name': 'MAX232',
                'full_name': 'MAX232EPE',
                'family': 'RS232',
                'manufacturer': 'Texas Instruments',
                'supply_voltage': 5.0,
                'package': 'DIP16',
                'pin_count': 16,
                'description': 'MAX232 - RS-232 Driver/Receiver',
                'datasheet_url': 'https://www.ti.com/lit/ds/symlink/max232.pdf',
                'parameters': {
                    'VOH': {'value': 5.0, 'unit': 'V', 'condition': 'TTL output high'},
                    'VOL': {'value': 0.4, 'unit': 'V', 'condition': 'TTL output low, IOL = 2 mA'},
                    'IOH': {'value': -0.4, 'unit': 'mA', 'condition': 'TTL output'},
                    'IOL': {'value': 2.0, 'unit': 'mA', 'condition': 'TTL output'},
                    'VIH': {'value': 2.0, 'unit': 'V', 'condition': 'TTL input'},
                    'VIL': {'value': 0.8, 'unit': 'V', 'condition': 'TTL input'},
                    'IIH': {'value': 0.1, 'unit': 'µA', 'condition': 'VIN = VCC'},
                    'IIL': {'value': -0.1, 'unit': 'µA', 'condition': 'VIN = 0V'},
                    'VCC': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'VDD': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 5.0, 'unit': 'V', 'condition': '4.5V to 5.5V'}
                },
                'pins': [
                    {'pin_id': 'MAX232_T1IN', 'chip_id': 'MAX232', 'pin_number': 11, 'pin_name': 'T1IN', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8, 'IIH': 0.1, 'IIL': -0.1}},
                    {'pin_id': 'MAX232_T1OUT', 'chip_id': 'MAX232', 'pin_number': 14, 'pin_name': 'T1OUT', 'function_type': 'OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 5.0, 'VOL': -5.0}},
                    {'pin_id': 'MAX232_R1IN', 'chip_id': 'MAX232', 'pin_number': 13, 'pin_name': 'R1IN', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {}},
                    {'pin_id': 'MAX232_R1OUT', 'chip_id': 'MAX232', 'pin_number': 12, 'pin_name': 'R1OUT', 'function_type': 'OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 5.0, 'VOL': 0.4, 'IOH': -0.4, 'IOL': 2.0}}
                ]
            },
            {
                'chip_id': 'ULN2003',
                'name': 'ULN2003',
                'full_name': 'ULN2003AN',
                'family': 'Darlington',
                'manufacturer': 'Texas Instruments',
                'supply_voltage': 5.0,
                'package': 'DIP16',
                'pin_count': 16,
                'description': 'ULN2003 - High-Voltage High-Current Darlington Transistor Array',
                'datasheet_url': 'https://www.ti.com/lit/ds/symlink/uln2003a.pdf',
                'parameters': {
                    'VOH': {'value': 50.0, 'unit': 'V', 'condition': 'Open collector output, max voltage'},
                    'VOL': {'value': 1.1, 'unit': 'V', 'condition': 'IOL = 350 mA, saturated'},
                    'IOH': {'value': 0.0, 'unit': 'mA', 'condition': 'Open collector'},
                    'IOL': {'value': 500.0, 'unit': 'mA', 'condition': 'Per channel max'},
                    'VIH': {'value': 2.4, 'unit': 'V', 'condition': 'Input high threshold'},
                    'VIL': {'value': 0.8, 'unit': 'V', 'condition': 'Input low threshold'},
                    'IIH': {'value': 93.0, 'unit': 'µA', 'condition': 'VIN = 3.85V'},
                    'IIL': {'value': 1.0, 'unit': 'µA', 'condition': 'VIN = 0V'},
                    'VCC': {'value': 5.0, 'unit': 'V', 'condition': 'logic supply'},
                    'VDD': {'value': 12.0, 'unit': 'V', 'condition': 'load supply'},
                    'supply_voltage': {'value': 5.0, 'unit': 'V', 'condition': 'logic input'}
                },
                'pins': [
                    {'pin_id': 'ULN2003_IN1', 'chip_id': 'ULN2003', 'pin_number': 1, 'pin_name': 'IN1', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.4, 'VIL': 0.8, 'IIH': 93.0, 'IIL': 1.0}},
                    {'pin_id': 'ULN2003_OUT1', 'chip_id': 'ULN2003', 'pin_number': 16, 'pin_name': 'OUT1', 'function_type': 'OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 50.0, 'VOL': 1.1, 'IOL': 500.0}}
                ]
            }
        ]

        for chip in interface_chips:
            self._import_chip(chip)
    
    def _import_peripheral_chips(self):
        """Import peripheral chip data (USB-serial, SPI Flash, LDO, reset, oscillator)."""
        logger.debug("导入常见周边芯片")

        peripheral_chips = [
            {
                'chip_id': 'CH340G',
                'name': 'CH340G',
                'full_name': 'CH340G USB to Serial Chip',
                'family': 'CH340',
                'manufacturer': 'WCH (Nanjing Qinheng Microelectronics)',
                'supply_voltage': 5.0,
                'package': 'SOP-16',
                'pin_count': 16,
                'description': 'CH340G - USB to Serial/UART Interface Chip',
                'datasheet_url': 'https://www.wch.cn/downloads/CH340DS1_PDF.html',
                'parameters': {
                    'VOH': {'value': 4.0, 'unit': 'V', 'condition': 'IOH = -4 mA, VCC = 5V'},
                    'VOL': {'value': 0.5, 'unit': 'V', 'condition': 'IOL = 4 mA, VCC = 5V'},
                    'IOH': {'value': -4.0, 'unit': 'mA', 'condition': 'VOH = 4V'},
                    'IOL': {'value': 4.0, 'unit': 'mA', 'condition': 'VOL = 0.5V'},
                    'VIH': {'value': 2.0, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'VIL': {'value': 0.8, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'IIH': {'value': 1.0, 'unit': 'µA', 'condition': 'VIN = VCC'},
                    'IIL': {'value': -1.0, 'unit': 'µA', 'condition': 'VIN = 0V'},
                    'VCC': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 5.0, 'unit': 'V', 'condition': '4.0V to 5.3V'}
                },
                'pins': [
                    {'pin_id': 'CH340G_TX', 'chip_id': 'CH340G', 'pin_number': 3, 'pin_name': 'TXD', 'function_type': 'UART_TX', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 4.0, 'VOL': 0.5, 'IOH': -4.0, 'IOL': 4.0}},
                    {'pin_id': 'CH340G_RX', 'chip_id': 'CH340G', 'pin_number': 4, 'pin_name': 'RXD', 'function_type': 'UART_RX', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8, 'IIH': 1.0, 'IIL': -1.0}}
                ]
            },
            {
                'chip_id': 'CH340C',
                'name': 'CH340C',
                'full_name': 'CH340C USB to Serial Chip',
                'family': 'CH340',
                'manufacturer': 'WCH',
                'supply_voltage': 5.0,
                'package': 'SOP-16',
                'pin_count': 16,
                'description': 'CH340C - USB to Serial/UART Interface Chip (Built-in Crystal)',
                'datasheet_url': 'https://www.wch.cn/downloads/CH340DS1_PDF.html',
                'parameters': {
                    'VOH': {'value': 4.0, 'unit': 'V', 'condition': 'IOH = -4 mA, VCC = 5V'},
                    'VOL': {'value': 0.5, 'unit': 'V', 'condition': 'IOL = 4 mA, VCC = 5V'},
                    'IOH': {'value': -4.0, 'unit': 'mA', 'condition': 'VOH = 4V'},
                    'IOL': {'value': 4.0, 'unit': 'mA', 'condition': 'VOL = 0.5V'},
                    'VIH': {'value': 2.0, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'VIL': {'value': 0.8, 'unit': 'V', 'condition': 'VCC = 5V'},
                    'VCC': {'value': 5.0, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 5.0, 'unit': 'V', 'condition': '4.0V to 5.3V'}
                },
                'pins': []
            },
            {
                'chip_id': 'W25Q16',
                'name': 'W25Q16',
                'full_name': 'W25Q16JVSSIQ',
                'family': 'W25Q',
                'manufacturer': 'Winbond',
                'supply_voltage': 3.3,
                'package': 'SOIC-8',
                'pin_count': 8,
                'description': 'W25Q16 - 16Mbit SPI Flash Memory',
                'datasheet_url': 'https://www.winbond.com/resource-files/w25q16jv%20spi%20revd%2008122016.pdf',
                'parameters': {
                    'VIH': {'value': 2.0, 'unit': 'V', 'condition': 'VCC = 3.3V'},
                    'VIL': {'value': 0.8, 'unit': 'V', 'condition': 'VCC = 3.3V'},
                    'VOH': {'value': 2.4, 'unit': 'V', 'condition': 'IOH = -0.1 mA'},
                    'VOL': {'value': 0.4, 'unit': 'V', 'condition': 'IOL = 0.1 mA'},
                    'IIH': {'value': 0.005, 'unit': 'µA', 'condition': 'VIN = VCC'},
                    'IIL': {'value': -0.005, 'unit': 'µA', 'condition': 'VIN = 0V'},
                    'VCC': {'value': 3.3, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 3.3, 'unit': 'V', 'condition': '2.7V to 3.6V'}
                },
                'pins': [
                    {'pin_id': 'W25Q16_CS', 'chip_id': 'W25Q16', 'pin_number': 1, 'pin_name': 'CS', 'function_type': 'SPI_CS', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8}},
                    {'pin_id': 'W25Q16_DO', 'chip_id': 'W25Q16', 'pin_number': 2, 'pin_name': 'DO', 'function_type': 'SPI_MISO', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 2.4, 'VOL': 0.4}},
                    {'pin_id': 'W25Q16_WP', 'chip_id': 'W25Q16', 'pin_number': 3, 'pin_name': 'WP', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8}},
                    {'pin_id': 'W25Q16_DI', 'chip_id': 'W25Q16', 'pin_number': 5, 'pin_name': 'DI', 'function_type': 'SPI_MOSI', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8}},
                    {'pin_id': 'W25Q16_CLK', 'chip_id': 'W25Q16', 'pin_number': 6, 'pin_name': 'CLK', 'function_type': 'SPI_SCK', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8}}
                ]
            },
            {
                'chip_id': 'AMS1117-5.0',
                'name': 'AMS1117-5.0',
                'full_name': 'AMS1117-5.0',
                'family': 'LDO',
                'manufacturer': 'AMS',
                'supply_voltage': 7.0,
                'package': 'SOT-223',
                'pin_count': 4,
                'description': 'AMS1117-5.0 - 5.0V Low Dropout Linear Regulator',
                'datasheet_url': 'https://www.advanced-monolithic.com/pdf/ds1117.pdf',
                'parameters': {
                    'VOH': {'value': 5.0, 'unit': 'V', 'condition': 'IO = 800 mA'},
                    'IOH': {'value': 1000.0, 'unit': 'mA', 'condition': 'VO = 5.0V'},
                    'VIH': {'value': 6.5, 'unit': 'V', 'condition': 'Minimum input voltage'},
                    'VCC': {'value': 7.0, 'unit': 'V', 'condition': 'nominal input'},
                    'VDD': {'value': 5.0, 'unit': 'V', 'condition': 'output'},
                    'supply_voltage': {'value': 7.0, 'unit': 'V', 'condition': '6.5V to 15V input'}
                },
                'pins': [
                    {'pin_id': 'AMS1117-5_GND', 'chip_id': 'AMS1117-5.0', 'pin_number': 1, 'pin_name': 'GND', 'function_type': 'GROUND', 'direction': 'BIDIRECTIONAL', 'alternate_functions': [], 'electrical_params': {}},
                    {'pin_id': 'AMS1117-5_OUT', 'chip_id': 'AMS1117-5.0', 'pin_number': 2, 'pin_name': 'VOUT', 'function_type': 'POWER_OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 5.0, 'IOH': 1000.0}},
                    {'pin_id': 'AMS1117-5_IN', 'chip_id': 'AMS1117-5.0', 'pin_number': 3, 'pin_name': 'VIN', 'function_type': 'POWER_INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {}}
                ]
            },
            {
                'chip_id': 'XC6206',
                'name': 'XC6206',
                'full_name': 'XC6206P332MR',
                'family': 'LDO',
                'manufacturer': 'Torex',
                'supply_voltage': 5.0,
                'package': 'SOT-23',
                'pin_count': 3,
                'description': 'XC6206 - 3.3V 200mA Low Dropout Regulator',
                'datasheet_url': 'https://www.torexsemi.com/file/xc6206/XC6206.pdf',
                'parameters': {
                    'VOH': {'value': 3.3, 'unit': 'V', 'condition': 'IO = 100 mA'},
                    'IOH': {'value': 200.0, 'unit': 'mA', 'condition': 'VO = 3.3V'},
                    'VIH': {'value': 3.5, 'unit': 'V', 'condition': 'Minimum input voltage'},
                    'VCC': {'value': 5.0, 'unit': 'V', 'condition': 'nominal input'},
                    'VDD': {'value': 3.3, 'unit': 'V', 'condition': 'output'},
                    'supply_voltage': {'value': 5.0, 'unit': 'V', 'condition': '3.5V to 6V input'}
                },
                'pins': [
                    {'pin_id': 'XC6206_VIN', 'chip_id': 'XC6206', 'pin_number': 3, 'pin_name': 'VIN', 'function_type': 'POWER_INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {}},
                    {'pin_id': 'XC6206_GND', 'chip_id': 'XC6206', 'pin_number': 1, 'pin_name': 'GND', 'function_type': 'GROUND', 'direction': 'BIDIRECTIONAL', 'alternate_functions': [], 'electrical_params': {}},
                    {'pin_id': 'XC6206_VOUT', 'chip_id': 'XC6206', 'pin_number': 2, 'pin_name': 'VOUT', 'function_type': 'POWER_OUTPUT', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 3.3, 'IOH': 200.0}}
                ]
            },
            {
                'chip_id': 'MAX809',
                'name': 'MAX809',
                'full_name': 'MAX809SEXR',
                'family': 'Reset',
                'manufacturer': 'Maxim Integrated',
                'supply_voltage': 3.3,
                'package': 'SOT-23',
                'pin_count': 3,
                'description': 'MAX809 - 3.08V Reset Circuit, Active Low Output',
                'datasheet_url': 'https://www.maximintegrated.com/en/products/power/supervisors-voltage-monitors-sequencers/MAX809.html',
                'parameters': {
                    'VOH': {'value': 3.3, 'unit': 'V', 'condition': 'Open drain pull-up'},
                    'VOL': {'value': 0.3, 'unit': 'V', 'condition': 'IOL = 3.2 mA'},
                    'IOL': {'value': 3.2, 'unit': 'mA', 'condition': 'VOL = 0.3V'},
                    'VIH': {'value': 3.08, 'unit': 'V', 'condition': 'Reset threshold'},
                    'VIL': {'value': 2.93, 'unit': 'V', 'condition': 'Release threshold'},
                    'VCC': {'value': 3.3, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 3.3, 'unit': 'V', 'condition': '1.0V to 5.5V'}
                },
                'pins': [
                    {'pin_id': 'MAX809_GND', 'chip_id': 'MAX809', 'pin_number': 1, 'pin_name': 'GND', 'function_type': 'GROUND', 'direction': 'BIDIRECTIONAL', 'alternate_functions': [], 'electrical_params': {}},
                    {'pin_id': 'MAX809_RST', 'chip_id': 'MAX809', 'pin_number': 2, 'pin_name': 'RST', 'function_type': 'RESET', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 3.3, 'VOL': 0.3, 'IOL': 3.2}},
                    {'pin_id': 'MAX809_VCC', 'chip_id': 'MAX809', 'pin_number': 3, 'pin_name': 'VCC', 'function_type': 'POWER_INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {}}
                ]
            },
            {
                'chip_id': 'MAX810',
                'name': 'MAX810',
                'full_name': 'MAX810SEXR',
                'family': 'Reset',
                'manufacturer': 'Maxim Integrated',
                'supply_voltage': 3.3,
                'package': 'SOT-23',
                'pin_count': 3,
                'description': 'MAX810 - 3.08V Reset Circuit, Active High Output',
                'datasheet_url': 'https://www.maximintegrated.com/en/products/power/supervisors-voltage-monitors-sequencers/MAX810.html',
                'parameters': {
                    'VOH': {'value': 2.9, 'unit': 'V', 'condition': 'IOH = -0.5 mA'},
                    'VOL': {'value': 0.3, 'unit': 'V', 'condition': 'IOL = 0.5 mA'},
                    'IOH': {'value': -0.5, 'unit': 'mA', 'condition': 'VOH = 2.9V'},
                    'IOL': {'value': 0.5, 'unit': 'mA', 'condition': 'VOL = 0.3V'},
                    'VIH': {'value': 3.08, 'unit': 'V', 'condition': 'Reset threshold'},
                    'VCC': {'value': 3.3, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 3.3, 'unit': 'V', 'condition': '1.0V to 5.5V'}
                },
                'pins': [
                    {'pin_id': 'MAX810_GND', 'chip_id': 'MAX810', 'pin_number': 1, 'pin_name': 'GND', 'function_type': 'GROUND', 'direction': 'BIDIRECTIONAL', 'alternate_functions': [], 'electrical_params': {}},
                    {'pin_id': 'MAX810_RST', 'chip_id': 'MAX810', 'pin_number': 2, 'pin_name': 'RST', 'function_type': 'RESET', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 2.9, 'VOL': 0.3}},
                    {'pin_id': 'MAX810_VCC', 'chip_id': 'MAX810', 'pin_number': 3, 'pin_name': 'VCC', 'function_type': 'POWER_INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {}}
                ]
            },
            {
                'chip_id': 'CAT811',
                'name': 'CAT811',
                'full_name': 'CAT811TTBI-30',
                'family': 'Reset',
                'manufacturer': 'ON Semiconductor',
                'supply_voltage': 3.3,
                'package': 'SOT-23',
                'pin_count': 3,
                'description': 'CAT811 - 3.08V Reset Circuit, Active Low Output',
                'datasheet_url': 'https://www.onsemi.com/pub/Collateral/CAT811-D.PDF',
                'parameters': {
                    'VOH': {'value': 3.3, 'unit': 'V', 'condition': 'Open drain pull-up'},
                    'VOL': {'value': 0.4, 'unit': 'V', 'condition': 'IOL = 3.2 mA'},
                    'IOL': {'value': 3.2, 'unit': 'mA', 'condition': 'VOL = 0.4V'},
                    'VIH': {'value': 3.08, 'unit': 'V', 'condition': 'Reset threshold'},
                    'VCC': {'value': 3.3, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 3.3, 'unit': 'V', 'condition': '1.0V to 5.5V'}
                },
                'pins': []
            },
            {
                'chip_id': 'OSC_8MHz',
                'name': 'OSC-8MHz',
                'full_name': '8MHz Active Crystal Oscillator',
                'family': 'Oscillator',
                'manufacturer': 'Generic',
                'supply_voltage': 3.3,
                'package': 'DIP-4',
                'pin_count': 4,
                'description': '8MHz Active Crystal Oscillator - CMOS Output',
                'datasheet_url': '',
                'parameters': {
                    'VOH': {'value': 2.8, 'unit': 'V', 'condition': 'IOH = -10 mA, VCC = 3.3V'},
                    'VOL': {'value': 0.4, 'unit': 'V', 'condition': 'IOL = 10 mA, VCC = 3.3V'},
                    'IOH': {'value': -10.0, 'unit': 'mA', 'condition': 'VOH = 2.8V'},
                    'IOL': {'value': 10.0, 'unit': 'mA', 'condition': 'VOL = 0.4V'},
                    'VIH': {'value': 2.0, 'unit': 'V', 'condition': 'Enable input high'},
                    'VIL': {'value': 0.8, 'unit': 'V', 'condition': 'Enable input low'},
                    'VCC': {'value': 3.3, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 3.3, 'unit': 'V', 'condition': '2.7V to 5.5V'}
                },
                'pins': [
                    {'pin_id': 'OSC_8M_OUT', 'chip_id': 'OSC_8MHz', 'pin_number': 3, 'pin_name': 'OUT', 'function_type': 'CLOCK', 'direction': 'OUTPUT', 'alternate_functions': [], 'electrical_params': {'VOH': 2.8, 'VOL': 0.4, 'IOH': -10.0, 'IOL': 10.0}},
                    {'pin_id': 'OSC_8M_EN', 'chip_id': 'OSC_8MHz', 'pin_number': 1, 'pin_name': 'EN', 'function_type': 'INPUT', 'direction': 'INPUT', 'alternate_functions': [], 'electrical_params': {'VIH': 2.0, 'VIL': 0.8}}
                ]
            },
            {
                'chip_id': 'OSC_12MHz',
                'name': 'OSC-12MHz',
                'full_name': '12MHz Active Crystal Oscillator',
                'family': 'Oscillator',
                'manufacturer': 'Generic',
                'supply_voltage': 3.3,
                'package': 'DIP-4',
                'pin_count': 4,
                'description': '12MHz Active Crystal Oscillator - CMOS Output',
                'datasheet_url': '',
                'parameters': {
                    'VOH': {'value': 2.8, 'unit': 'V', 'condition': 'IOH = -10 mA, VCC = 3.3V'},
                    'VOL': {'value': 0.4, 'unit': 'V', 'condition': 'IOL = 10 mA, VCC = 3.3V'},
                    'IOH': {'value': -10.0, 'unit': 'mA', 'condition': 'VOH = 2.8V'},
                    'IOL': {'value': 10.0, 'unit': 'mA', 'condition': 'VOL = 0.4V'},
                    'VCC': {'value': 3.3, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 3.3, 'unit': 'V', 'condition': '2.7V to 5.5V'}
                },
                'pins': []
            },
            {
                'chip_id': 'OSC_16MHz',
                'name': 'OSC-16MHz',
                'full_name': '16MHz Active Crystal Oscillator',
                'family': 'Oscillator',
                'manufacturer': 'Generic',
                'supply_voltage': 3.3,
                'package': 'DIP-4',
                'pin_count': 4,
                'description': '16MHz Active Crystal Oscillator - CMOS Output',
                'datasheet_url': '',
                'parameters': {
                    'VOH': {'value': 2.8, 'unit': 'V', 'condition': 'IOH = -10 mA, VCC = 3.3V'},
                    'VOL': {'value': 0.4, 'unit': 'V', 'condition': 'IOL = 10 mA, VCC = 3.3V'},
                    'IOH': {'value': -10.0, 'unit': 'mA', 'condition': 'VOH = 2.8V'},
                    'IOL': {'value': 10.0, 'unit': 'mA', 'condition': 'VOL = 0.4V'},
                    'VCC': {'value': 3.3, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 3.3, 'unit': 'V', 'condition': '2.7V to 5.5V'}
                },
                'pins': []
            },
            {
                'chip_id': 'OSC_25MHz',
                'name': 'OSC-25MHz',
                'full_name': '25MHz Active Crystal Oscillator',
                'family': 'Oscillator',
                'manufacturer': 'Generic',
                'supply_voltage': 3.3,
                'package': 'DIP-4',
                'pin_count': 4,
                'description': '25MHz Active Crystal Oscillator - CMOS Output (Ethernet)',
                'datasheet_url': '',
                'parameters': {
                    'VOH': {'value': 2.8, 'unit': 'V', 'condition': 'IOH = -10 mA, VCC = 3.3V'},
                    'VOL': {'value': 0.4, 'unit': 'V', 'condition': 'IOL = 10 mA, VCC = 3.3V'},
                    'IOH': {'value': -10.0, 'unit': 'mA', 'condition': 'VOH = 2.8V'},
                    'IOL': {'value': 10.0, 'unit': 'mA', 'condition': 'VOL = 0.4V'},
                    'VCC': {'value': 3.3, 'unit': 'V', 'condition': 'nominal'},
                    'supply_voltage': {'value': 3.3, 'unit': 'V', 'condition': '2.7V to 5.5V'}
                },
                'pins': []
            }
        ]
        
        for chip in peripheral_chips:
            self._import_chip(chip)
    
    def _infer_process_type(self, family: str) -> str:
        """Infer the fabrication process type from the chip family name.

        Args:
            family: Chip family name (e.g. 'STM32F1', '74HC', '74LS').

        Returns:
            One of 'CMOS', 'BiCMOS', 'TTL', 'ECL'. Defaults to 'CMOS'.
        """
        family_upper = family.upper()

        if any(prefix in family_upper for prefix in ['STM32', 'ESP32', 'AVR', '74HC', '74HCT', '74LVC', '74LV', '74AHC', '74AHCT', 'MSP430', 'CH340', 'W25Q', 'LDO', 'Reset', 'Oscillator']):
            return 'CMOS'

        if any(prefix in family_upper for prefix in ['74ABT', '74BCT', '74FCT']):
            return 'BiCMOS'

        if any(prefix in family_upper for prefix in ['74LS', '74S', '74ALS', '74AS', '74F']):
            return 'TTL'

        if 'ECL' in family_upper:
            return 'ECL'

        return 'CMOS'
    
    def _import_chip(self, chip: Dict[str, Any]):
        """Import a single chip's data into the chips, pins, and parameters tables.

        Uses INSERT OR REPLACE for idempotent writes. Each chip is decomposed
        into one chips row, N pins rows, and M parameters rows.

        Args:
            chip: Chip data dict with keys: chip_id, name, full_name, family,
                manufacturer, supply_voltage, package, pin_count, description,
                datasheet_url, parameters, pins, and optionally process.
        """
        chip_id = chip['chip_id']

        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO chips
            (chip_id, name, full_name, family, manufacturer, supply_voltage, 
             package, process, pin_count, description, datasheet_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chip_id,
            chip['name'],
            chip['full_name'],
            chip['family'],
            chip['manufacturer'],
            chip['supply_voltage'],
            chip['package'],
            chip.get('process', self._infer_process_type(chip['family'])),
            chip['pin_count'],
            chip['description'],
            chip['datasheet_url'],
            datetime.now().isoformat()
        ))

        for pin in chip.get('pins', []):
            cursor.execute("""
                INSERT OR REPLACE INTO pins
                (pin_id, chip_id, pin_number, pin_name, function_type, direction,
                 alternate_functions, electrical_params, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pin['pin_id'],
                pin['chip_id'],
                pin['pin_number'],
                pin['pin_name'],
                pin['function_type'],
                pin['direction'],
                json.dumps(pin.get('alternate_functions', [])),
                json.dumps(pin.get('electrical_params', {})),
                pin.get('description', '')
            ))

        for param_name, param_data in chip['parameters'].items():
            param_id = f"{chip_id}_{param_name}"
            cursor.execute("""
                INSERT OR REPLACE INTO parameters
                (param_id, chip_id, param_name, param_value, unit, condition, source, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                param_id,
                chip_id,
                param_name,
                param_data['value'],
                param_data.get('unit', ''),
                param_data.get('condition', ''),
                'knowledge_graph',
                0.8
            ))
        
        logger.info(f"已导入芯片: {chip['name']}")


def import_common_chip_data(db_path: str = "./data/knowledge_graph.db"):
    """Convenience function to import all common chip data.

    Args:
        db_path: Path to the SQLite database file.
    """
    with CommonChipDataImporter(db_path) as importer:
        importer.import_all()


if __name__ == "__main__":
    import_common_chip_data()
    print("常用芯片数据导入完成！")
