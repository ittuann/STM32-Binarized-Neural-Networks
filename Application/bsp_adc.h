#pragma once

#include "adc.h"

extern volatile float ADCVrefintProportion;

extern uint16_t ADCx_get_chx_value(ADC_HandleTypeDef *ADCx, uint32_t ch);
extern void ADC_vrefint_init(void);
extern float ADC_get_STM32Temprate(void);
