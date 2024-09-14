/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "spi.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdint.h>
#include <stdio.h>
#include "stm32f4xx.h"

#include "mlp_ebnn.h"
#include "mlp_ebnn_mnist_data.h"
#include "mlp_ebnn_data.h"


#include "arm_math.h"
#include "bsp_adc.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
#define REDIRECT2SWV
#define PUTCHAR_PROTOTYPE __attribute__((weak)) int fputc(int ch, FILE *f)

#ifdef REDIRECT2SWV 
#define ITM_Port8(n)    (*((volatile unsigned char *)(0xE0000000+4*n)))
#define ITM_Port32(n)   (*((volatile unsigned long *)(0xE0000000+4*n)))

PUTCHAR_PROTOTYPE
{
	while (ITM_Port32(0) == 0);
	ITM_Port8(0) = ch;
    return (ch);
}
#else
PUTCHAR_PROTOTYPE
{
//	HAL_UART_Transmit_IT(&huart1, (uint8_t *)&ch, 1);
    HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, 0xFFFF);
	return (ch);
}
#endif



volatile uint32_t executionTime, startTime;
volatile float executionAveTime;
float temprate;


// 输入数据
const int batch_size = 20;
const int input_dim = 784;
const int output_dim = 10;

// 中间层输出
int fc_output[batch_size * output_dim];
float bn_output[batch_size * output_dim];
float softmax_output[batch_size * output_dim];
int max_softmax_output[batch_size];
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART1_UART_Init();
  MX_ADC1_Init();
  MX_SPI1_Init();
  /* USER CODE BEGIN 2 */
  HAL_Delay(100);	// Wait for all peripherals to power up.
//	ADC_vrefint_init();

  printf("SystemCoreClock: %d\n", SystemCoreClock);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
    while (1)
    {
        /*
            计算时间
        */
        startTime = HAL_GetTick();		// Timing, use SysTick. Also an option to use DWT/DMY.
        // 全连接层推理
        binary_bitwise_fully_connected_inference_uint8(test_data, batch_size, input_dim, output_dim, binarize_fc1_w, binarize_fc1_b, fc_output);

        // 批量归一化推理
        // batch_normalization_inference(fc_output, batch_size, output_dim, binarize_fc1_bn_gamma, binarize_fc1_bn_beta, binarize_fc1_bn_mean, binarize_fc1_bn_var, 1e-5, bn_output);
        batch_normalization_inference_std_int(fc_output, batch_size, output_dim, binarize_fc1_bn_gamma, binarize_fc1_bn_beta, binarize_fc1_bn_mean, binarize_fc1_bn_std, bn_output);

        // Softmax 推理
        softmax_inference(bn_output, batch_size, output_dim, softmax_output);
        // 最大概率索引
        max_softmax_inference(softmax_output, batch_size, output_dim, max_softmax_output);
        /*
            计算时间
        */
        executionTime = HAL_GetTick() - startTime;
        executionAveTime = executionTime / 20.0f;
        // 打印预测结果
        int16_t correct_predictions = 0;
        printf("Model Predictions vs Actual Labels:\n");
        for (int i = 0; i < batch_size; i++) {
            printf("Sample %2d: Predicted = %d, Actual = %d", i, max_softmax_output[i], test_labels[i]);
            if (max_softmax_output[i] == test_labels[i]) {
                printf(" (Correct)\n");
                correct_predictions++;
            } else {
                printf(" (Incorrect)\n");
            }
        }
        // 计算并输出准确率
        float accuracy = (float)correct_predictions / batch_size * 100.0f;
        printf("\nCorrect: %d; Incorrect: %d; Total: %d\n", correct_predictions, batch_size - correct_predictions, batch_size);
        printf("Accuracy: %.2f%%\n", accuracy);

        printf("Usage time: %d ms\n", executionTime);
        printf("Usage Ave time: %d ms\n", (int)(executionAveTime));
        
        HAL_GPIO_TogglePin(LED_GPIO_Port, LED_Pin);
        
//		temprate = ADC_get_STM32Temprate();
//		printf("MCU temperature: %d\n", (int)(temprate * 100));
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 12;
  RCC_OscInitStruct.PLL.PLLN = 96;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
