#pragma region Task1
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <locale.h>
//#include <stdio.h>
//
//int main() {
//	cudaDeviceProp prop;
//	setlocale(LC_ALL, "RUS");
//	int count;
//	cudaGetDeviceCount(&count);
//	for (int i = 0; i < count; i++) {
//		cudaGetDeviceProperties(&prop, i);
//		printf(" - Общая информация об устройстве %d - \n", i);
//		printf(" Имя: %s\n", prop.name);
//		printf(" Вычислительные возможности: %d.%d\n", prop.major, prop.minor);
//		printf(" Тактовая частота: %d\n", prop.clockRate);
//		printf(" Перекрытие копирования: ");
//		if (prop.deviceOverlap)
//			printf(" разрешено\n");
//		else printf("запрещено\n");
//		printf("Тайм-аут выполнения ядра: ");
//		if (prop.kernelExecTimeoutEnabled)
//			printf(" включен \n ");
//		else printf(" выключен \n ");
//		printf("-Информация о памяти для устройства %d - \n", i);
//		printf("Всего глобальной памяти: %ld\n", prop.totalGlobalMem);
//		printf("Всего константной памяти: %ld\n", prop.totalConstMem);
//		printf("Максимальный шаг: %ld\n", prop.memPitch);
//		printf("Выравнивание текстур: %ld\n", prop.textureAlignment);
//		printf("Инфо о мультипроцессорах для устройства %d - \n", i);
//		printf("Кол-во мультипроцессорах: %d\n", prop.multiProcessorCount);
//		printf("Разделяемая память на один МП: %ld\n", prop.sharedMemPerBlock);
//		printf("Количество регистров на один МП: %ld\n", prop.regsPerBlock);
//		printf("Количество нитей в варпе: %d\n", prop.warpSize);
//		printf("Максимальное количество нитей в блоке: %d\n", prop.maxThreadsPerBlock);
//		printf("Макс. кол-во нитей по измерениям: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
//		printf("Макс. размеры сетки: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
//		printf("\n");
//	}
//	char c = getchar();
//}

#pragma endregion

#pragma region Task2
//#include "cuda_runtime.h"               // Подключение заголовочного файла CUDA runtime API
//#include "device_launch_parameters.h"   // Подключение заголовочного файла для параметров запуска устройства
//#include <stdio.h>                      
//#include <stdlib.h> 
//
//// Определение ядра CUDA, которое будет выполняться на GPU
//__global__ void kernel() {
//    int tid = threadIdx.x;                          // Получение уникального идентификатора потока внутри блока
//    printf("Thread number %d\n", tid);              // Вывод номера текущего потока
//}
//
//int main(int argc, char* argv[])
//{
//    // Проверка, передан ли аргумент командной строки
//    if (argc != 2) {
//        fprintf(stderr, "Usage: %s <number_of_threads>\n", argv[0]);
//        return 1;
//    }
//
//    // Преобразование аргумента из строки в целое число
//    int N = atoi(argv[1]);
//
//    // Проверка, что N положительное и не превышает максимальное количество потоков в блоке
//    if (N <= 0 || N > 1024) { // Обычно максимальное количество потоков в блоке - 1024
//        fprintf(stderr, "Error: number_of_threads must be between 1 and 1024.\n");
//        return 1;
//    }
//
//    int* dev_a;                                      // Объявление указателя для памяти на устройстве (GPU)
//
//    // Выделение памяти на GPU для одного целого числа
//    cudaError_t err = cudaMalloc((void**)&dev_a, sizeof(int));
//    if (err != cudaSuccess) {
//        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
//        return 1;
//    }
//
//    // Запуск ядра CUDA с 1 блоком и N потоками в блоке
//    kernel <<<1, N >>> ();
//
//    err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
//        cudaFree(dev_a);
//        return 1;
//    }
//
//    // Синхронизация устройства, чтобы убедиться, что все потоки завершили выполнение
//    err = cudaDeviceSynchronize();
//    if (err != cudaSuccess) {
//        fprintf(stderr, "CUDA synchronization failed: %s\n", cudaGetErrorString(err));
//        cudaFree(dev_a);
//        return 1;
//    }
//
//    // Освобождение ранее выделенной памяти на устройстве GPU
//    cudaFree(dev_a);
//
//    getchar();                                       // Ожидание нажатия клавиши пользователем перед завершением программы
//    return 0;                                       // Возврат 0, указывающий на успешное завершение программы
//}
#pragma endregion

#pragma region Task3
//#include "cuda_runtime.h"               // Подключение заголовочного файла CUDA runtime API
//#include "device_launch_parameters.h"   // Подключение заголовочного файла для параметров запуска устройства
//#include <stdio.h>                      
//#include <stdlib.h>                     
//
//// Удаление определения N и M через препроцессор, так как теперь они будут задаваться из аргументов командной строки
//
//// Определение ядра CUDA, которое будет выполняться на GPU
//__global__ void kernel() {
//    int tid_block = blockIdx.x;       // Получение идентификатора блока по оси x
//    int tid_thread = threadIdx.x;     // Получение идентификатора потока внутри блока по оси x
//    printf("Block number %d\t Thread number %d\n", tid_block, tid_thread); // Вывод номера блока и потока
//}
//
//int main(int argc, char* argv[])
//{
//    // Проверка, переданы ли необходимые аргументы командной строки
//    if (argc != 3) {
//        fprintf(stderr, "Usage: %s <number_of_blocks> <number_of_threads>\n", argv[0]);
//        return 1;
//    }
//
//    // Преобразование аргументов из строк в целые числа
//    int M = atoi(argv[1]); // Количество блоков
//    int N = atoi(argv[2]); // Количество потоков в блоке
//
//    if (N <= 0 || N > 1024) { // Обычно максимальное количество потоков в блоке - 1024
//        fprintf(stderr, "Error: number_of_threads must be between 1 and 1024.\n");
//        return 1;
//    }
//
//    int* dev_a; // Объявление указателя для памяти на устройстве (GPU)
//
//    // Выделение памяти на GPU для одного целого числа
//    cudaError_t err = cudaMalloc((void**)&dev_a, sizeof(int));
//    if (err != cudaSuccess) {
//        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
//        return 1;
//    }
//
//    // Запуск ядра CUDA с M блоками и N потоками в каждом блоке
//    kernel<<<M, N>>>();
//
//    // Проверка на ошибки после запуска ядра
//    err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
//        cudaFree(dev_a);
//        return 1;
//    }
//
//    // Синхронизация устройства, чтобы убедиться, что все потоки завершили выполнение
//    err = cudaDeviceSynchronize();
//    if (err != cudaSuccess) {
//        fprintf(stderr, "CUDA synchronization failed: %s\n", cudaGetErrorString(err));
//        cudaFree(dev_a);
//        return 1;
//    }
//
//    // Освобождение ранее выделенной памяти на устройстве GPU
//    cudaFree(dev_a);
//
//    // Ожидание нажатия клавиши пользователем перед завершением программы
//    getchar();
//
//    return 0; // Возврат 0, указывающий на успешное завершение программы
//}
#pragma endregion

#pragma region Task4
//#include <stdio.h>                       // Подключение стандартной библиотеки ввода-вывода
//#include <cuda_runtime.h>                // Подключение заголовочного файла CUDA runtime API
//#include <cuda.h>                        // Подключение заголовочного файла CUDA
//#include "device_launch_parameters.h"    // Подключение заголовочного файла для параметров запуска устройства
//#include <locale.h>                      // Подключение заголовочного файла для локализации
//
//// Размер блока выбран как степень 2 для эффективной редукции
//#define BLOCK_SIZE 256                    // Определение размера блока (количество потоков в блоке)
//
//// Определение ядра CUDA для редукции суммы единиц
//__global__ void sumOnesReduction(float* output, int n) {
//    __shared__ float sdata[BLOCK_SIZE];   // Объявление shared памяти для хранения данных каждого блока
//
//    // Вычисляем глобальный индекс потока
//    int tid = threadIdx.x;                                // Индекс потока внутри блока
//    int i = blockIdx.x * blockDim.x + threadIdx.x;        // Глобальный индекс потока в сетке
//
//    // Инициализируем shared память значениями 1.0f или 0.0f в зависимости от индекса
//    sdata[tid] = (i < n) ? 1.0f : 0.0f;                   // Если индекс внутри диапазона, присваиваем 1.0, иначе 0.0
//    __syncthreads();                                      // Синхронизация потоков внутри блока
//
//    // Выполняем редукцию (суммирование) в shared памяти
//    for (int s = blockDim.x / 2; s > 0; s >>= 1) {        // Итерации для редукции (делим шаг пополам на каждой итерации)
//        if (tid < s) {                                     // Если индекс потока меньше текущего шага
//            sdata[tid] += sdata[tid + s];                 // Суммируем значение текущего потока с потоком на расстоянии s
//        }
//        __syncthreads();                                  // Синхронизация потоков после каждой стадии редукции
//    }
//
//    // Записываем результат редукции для данного блока в глобальную память
//    if (tid == 0) {                                        // Только поток с индексом 0 записывает результат
//        output[blockIdx.x] = sdata[0];                    // Запись суммы всех единиц данного блока
//    }
//}
//
//int main(int argc, char* argv[]) {
//    setlocale(LC_ALL, "RUS");                             // Установка локали для корректного отображения русских символов
//
//    // Проверка, передан ли необходимый аргумент командной строки
//    if (argc != 2) {
//        printf("Использование: %s <количество_элементов>\n", argv[0]);
//        return -1;                                        // Завершение программы с ошибкой, если аргумент не передан
//    }
//
//    // Создание событий CUDA для измерения времени выполнения ядра
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);                               // Создание события начала
//    cudaEventCreate(&stop);                                // Создание события окончания
//
//    // Преобразование аргумента командной строки в целое число
//    int numElements = atoi(argv[1]);
//    printf("Заданное количество элементов: %d\n", numElements);
//
//    // Вычисляем количество блоков для первого прохода редукции
//    int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE; // Округление вверх для покрытия всех элементов
//    size_t size = numBlocks * sizeof(float);                       // Размер памяти для хранения промежуточных сумм
//
//    // Выделяем память для промежуточных результатов на хосте (CPU)
//    float* h_Output = (float*)malloc(size);
//    if (h_Output == NULL) {
//        fprintf(stderr, "Ошибка выделения памяти на хосте.\n");
//        return -1;
//    }
//
//    // Выделяем память для промежуточных результатов на устройстве (GPU)
//    float* d_Output = NULL;
//    cudaError_t err = cudaMalloc((void**)&d_Output, size);
//    if (err != cudaSuccess) {                                     // Проверка успешности выделения памяти
//        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
//        free(h_Output);
//        return -1;
//    }
//
//    // Запись события начала
//    cudaEventRecord(start, 0);
//
//    // Запуск ядра CUDA с numBlocks блоками и BLOCK_SIZE потоками в каждом блоке
//    sumOnesReduction <<<numBlocks, BLOCK_SIZE >>> (d_Output, numElements);
//
//    // Проверка на ошибки после запуска ядра
//    err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        fprintf(stderr, "Запуск ядра не удался: %s\n", cudaGetErrorString(err));
//        cudaFree(d_Output);
//        free(h_Output);
//        return -1;
//    }
//
//    // Копируем промежуточные результаты из устройства на хост
//    cudaMemcpy(h_Output, d_Output, size, cudaMemcpyDeviceToHost);
//
//    // Суммируем финальные результаты на CPU
//    float finalSum = 0.0f;
//    for (int i = 0; i < numBlocks; i++) {
//        finalSum += h_Output[i];                                // Суммирование всех промежуточных сумм
//    }
//
//    // Запись события окончания
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);                                   // Ожидание завершения всех событий
//
//    // Вычисление времени выполнения ядра
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    printf("Время выполнения ядра: %f миллисекунд\n", milliseconds);
//    printf("Сумма %d единиц равна: %.0f\n", numElements, finalSum);
//
//    // Освобождаем выделенную память
//    cudaFree(d_Output);                                          // Освобождение памяти на устройстве (GPU)
//    free(h_Output);                                              // Освобождение памяти на хосте (CPU)
//
//    // Ожидание нажатия клавиши пользователем перед завершением программы
//    printf("Нажмите любую клавишу для выхода...\n");
//    getchar();
//
//    return 0;                                                    // Успешное завершение программы
//}
#pragma endregion

#pragma region Task5
//#include <stdio.h>                       
//#include <stdlib.h>                      
//#include <math.h>                        
//#include <cuda.h>                        // Заголовочный файл для CUDA
//#include <cuda_runtime.h>                // Заголовочный файл для CUDA Runtime API
//#include "device_launch_parameters.h"    // Параметры запуска устройства CUDA
//#include <locale.h>                      
//#include <time.h>                        
//
//#define THREADS_PER_BLOCK 256             // Определение количества потоков в одном блоке
//#define BLOCKS 256                        // Определение количества блоков
//
//// Функция для интегрирования на CPU
//double f(double x) {
//    return 0.06 * x * x * x + 0.3 * x * x - 8.0 * x + 110.0;
//}
//
//// Последовательное вычисление интеграла на CPU
//double integrateCPU(long long num_intervals, double a, double b) {
//    double h = (b - a) / num_intervals;     // Вычисление шага интегрирования
//    double sum = 0.0;                       // Инициализация суммы
//
//    // Цикл по всем интервалам для вычисления суммы значений функции
//    for (long long i = 0; i < num_intervals; i++) {
//        double x = a + (i + 0.5) * h;       // Вычисление точки середины текущего интервала
//        sum += f(x);                        // Суммирование значений функции
//    }
//
//    return h * sum;                         // Умножение суммы на шаг интегрирования для получения результата
//}
//
//// Функция для интегрирования на GPU (используется внутри ядра)
//__device__ float fGPU(float x) {
//    return 0.06f * x * x * x + 0.3f * x * x - 8.0f * x + 110.0f;
//}
//
//// Оптимизированное ядро CUDA для параллельного интегрирования
//__global__ void integrateKernel(float* result_d, float h, int num_intervals, float a) {
//    __shared__ float shared_sum[THREADS_PER_BLOCK];  // Shared память для хранения промежуточных сумм каждого блока
//
//    int tid = threadIdx.x;                            // Индекс текущего потока внутри блока
//    int total_threads = blockDim.x * gridDim.x;       // Общее количество потоков в сетке
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Глобальный индекс потока
//
//    float local_sum = 0.0f;                           // Локальная сумма для каждого потока
//
//    // Цикл по интервалам с шагом, равным общему количеству потоков
//    for (int i = idx; i < num_intervals; i += total_threads) {
//        float x = a + (i + 0.5f) * h;                 // Вычисление точки середины интервала
//        local_sum += fGPU(x);                         // Суммирование значений функции
//    }
//
//    shared_sum[tid] = local_sum;                       // Сохранение локальной суммы в shared память
//    __syncthreads();                                   // Синхронизация всех потоков внутри блока
//
//    // Редукция (суммирование) внутри блока
//    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//        if (tid < stride) {
//            shared_sum[tid] += shared_sum[tid + stride]; // Суммирование пар элементов
//        }
//        __syncthreads();                               // Синхронизация после каждой стадии редукции
//    }
//
//    // Добавление суммы блока к глобальной сумме с использованием атомарной операции
//    if (tid == 0) {
//        atomicAdd(result_d, shared_sum[0]);            // Атомарное добавление результата блока к глобальной сумме
//    }
//}
//
//clock_t t;                                            // Переменная для измерения времени
//double cpu_result;                                    // Результат интегрирования на CPU
//
//int main(int argc, char* argv[]) {
//    setlocale(LC_ALL, "RUS");                         // Установка локали для корректного отображения русских символов
//
//    // Проверка наличия необходимого аргумента командной строки
//    if (argc != 2) {
//        printf("Использование: %s <число_отрезков>\n", argv[0]); // Инструкция по использованию программы
//        return -1;                                    // Завершение программы с ошибкой
//    }
//
//    long long num_intervals = atoll(argv[1]);         // Преобразование аргумента командной строки в тип long long
//    printf("Число отрезков: %lld\n", num_intervals); // Вывод количества отрезков
//
//    float a = -10.0f;                                 // Начало интервала интегрирования
//    float b = 20.0f;                                  // Конец интервала интегрирования
//    float h = (b - a) / num_intervals;                // Вычисление шага интегрирования
//
//    // Последовательное вычисление интеграла на CPU
//    t = clock();                                      // Начало отсчёта времени
//    cpu_result = integrateCPU(num_intervals, a, b);   // Вычисление интеграла на CPU
//    t = clock() - t;                                  // Конец отсчёта времени
//    double time_seq = (double)t * 1000 / CLOCKS_PER_SEC; // Перевод времени в миллисекунды
//
//    printf("Результат последовательного вычисления: %.10f\n", cpu_result); // Вывод результата CPU
//    printf("Время последовательного вычисления: %f миллисекунд\n", time_seq); // Вывод времени CPU
//
//    // Параллельное вычисление интеграла на GPU
//    float* result_d;                                   // Указатель на результат на устройстве (GPU)
//    float result_h = 0.0f;                             // Результат на хосте (CPU)
//
//    cudaMalloc((void**)&result_d, sizeof(float));       // Выделение памяти на GPU для результата
//    cudaMemset(result_d, 0, sizeof(float));            // Инициализация результата нулём
//
//    cudaEvent_t start_par, stop_par;                    // События для измерения времени выполнения ядра
//    cudaEventCreate(&start_par);                        // Создание события начала
//    cudaEventCreate(&stop_par);                         // Создание события окончания
//
//    cudaEventRecord(start_par, 0);                      // Запись события начала
//    integrateKernel << < BLOCKS, THREADS_PER_BLOCK >> > (result_d, h, num_intervals, a); // Запуск ядра CUDA
//    cudaEventRecord(stop_par, 0);                       // Запись события окончания
//
//    cudaMemcpy(&result_h, result_d, sizeof(float), cudaMemcpyDeviceToHost); // Копирование результата с GPU на CPU
//    result_h *= h;                                       // Умножение на шаг интегрирования для получения итогового значения
//    cudaEventSynchronize(stop_par);                     // Ожидание завершения события окончания
//
//    float time_par;                                     // Время выполнения ядра
//    cudaEventElapsedTime(&time_par, start_par, stop_par); // Вычисление прошедшего времени в миллисекундах
//
//    printf("Результат параллельного вычисления: %.10f\n", result_h); // Вывод результата GPU
//    printf("Время параллельного вычисления: %f миллисекунд\n", time_par); // Вывод времени GPU
//    double acceleration = time_seq / time_par;             // Вычисление ускорения
//
//    printf("Ускорение: %.2f раз\n", acceleration);           // Вывод ускорения
//
//    // Проверка ошибок CUDA
//    cudaError_t error = cudaGetLastError();
//    if (error != cudaSuccess) {
//        printf("CUDA error: %s\n", cudaGetErrorString(error)); // Вывод сообщения об ошибке CUDA
//    }
//
//    // Освобождение ресурсов
//    cudaFree(result_d);                                   // Освобождение памяти на GPU
//    cudaEventDestroy(start_par);                          // Удаление события начала
//    cudaEventDestroy(stop_par);                           // Удаление события окончания
//
//    return 0;                                             // Успешное завершение программы
//}

#pragma endregion

#pragma region Task6
#include <stdio.h>                      // Библиотека для стандартного ввода-вывода
#include <stdlib.h>                     // Библиотека стандартных функций
#include <math.h>                       // Математические функции
#include <cuda.h>                       // Основные функции CUDA
#include <cuda_runtime.h>               // Исполняемые функции CUDA
#include "device_launch_parameters.h"   // Параметры запуска устройства CUDA
#include <locale.h>                     // Установка локали
#include <time.h>                       // Работа со временем

#define THREADS_PER_BLOCK 256           // Количество потоков в одном блоке
#define BLOCKS 256                      // Количество блоков в сетке

// Функция для интегрирования на CPU
double f(double x) {
    return 0.06 * x * x * x + 0.3 * x * x - 8.0 * x + 110.0;  // Вычисление значения функции в точке x
}

// Последовательное вычисление интеграла на CPU методом прямоугольников
double integrateCPU(int num_intervals, double a, double b) {
    double h = (b - a) / num_intervals;       // Шаг интегрирования
    double sum = 0.0;                         // Переменная для накопления суммы
    for (int i = 0; i < num_intervals; i++) { // Цикл по интервалам
        double x = a + (i + 0.5) * h;         // Точка середины текущего интервала
        sum += f(x);                          // Добавление значения функции в точке x к сумме
    }
    return h * sum;                           // Возвращаем значение интеграла
}

// Функция для интегрирования на GPU
__device__ double fGPU(double x) {
    return 0.06f * x * x * x + 0.3f * x * x - 8.0f * x + 110.0f;  // Вычисление значения функции на устройстве (GPU)
}

// Оптимизированное ядро CUDA для параллельного вычисления интеграла
__global__ void integrateKernel(double* result_d, double h, long long num_intervals, double a) {
    __shared__ double shared_sum[THREADS_PER_BLOCK]; // Общая память для сумм внутри блока

    int tid = threadIdx.x;                           // Локальный идентификатор потока в блоке
    int total_threads = blockDim.x * gridDim.x;      // Общее количество потоков во всей сетке
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный идентификатор потока

    double local_sum = 0.0f;                         // Локальная сумма для каждого потока
    for (long long i = idx; i < num_intervals; i += total_threads) { // Цикл по интервалам с шагом в общее число потоков
        double x = a + (i + 0.5f) * h;               // Точка середины текущего интервала
        local_sum += fGPU(x);                        // Добавление значения функции в точке x к локальной сумме
    }

    shared_sum[tid] = local_sum;                     // Сохранение локальной суммы в общую память
    __syncthreads();                                  // Синхронизация потоков внутри блока

    // Редукция сумм в пределах блока
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride]; // Суммирование пар элементов
        }
        __syncthreads();                              // Синхронизация после каждой итерации редукции
    }

    if (tid == 0) {
        atomicAdd(result_d, shared_sum[0]);           // Атомарное добавление суммы блока к общему результату
    }
}

clock_t t;                // Переменная для измерения времени
double cpu_result;        // Переменная для хранения результата на CPU

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "RUS");                         // Установка русской локали для корректного отображения

    long long num_intervals = atoll(argv[1]);         // Чтение числа интервалов из аргументов командной строки
    printf("Число отрезков: %lld\n", num_intervals);  // Вывод количества интервалов

    double a = -10.0f;                                // Левая граница интегрирования
    double b = 20.0f;                                 // Правая граница интегрирования
    double h = (b - a) / num_intervals;               // Вычисление шага интегрирования

    // Последовательное вычисление интеграла на CPU
    t = clock();                                      // Начало отсчёта времени
    cpu_result = integrateCPU(num_intervals, a, b);   // Вычисление интеграла на CPU
    t = clock() - t;                                  // Конец отсчёта времени
    double time_seq = (double)t * 1000 / CLOCKS_PER_SEC; // Перевод времени в миллисекунды

    printf("Результат последовательного вычисления: %.10f\n", cpu_result); // Вывод результата CPU
    printf("Время последовательного вычисления: %f миллисекунд\n", time_seq); // Вывод времени CPU

    // Параллельное вычисление интеграла на GPU
    double* result_d;                                 // Указатель на результат на устройстве (GPU)
    double result_h = 0.0f;                           // Переменная для хранения результата на хосте (CPU)

    cudaMalloc((void**)&result_d, sizeof(double));    // Выделение памяти на устройстве для результата
    cudaMemset(result_d, 0, sizeof(double));          // Инициализация выделенной памяти нулями

    cudaEvent_t start_par, stop_par;                  // Создание событий для измерения времени на GPU
    cudaEventCreate(&start_par);                      // Создание события начала
    cudaEventCreate(&stop_par);                       // Создание события окончания

    cudaEventRecord(start_par, 0);                    // Запись времени начала вычислений на GPU
    integrateKernel << <BLOCKS, THREADS_PER_BLOCK >> > (result_d, h, num_intervals, a); // Запуск ядра CUDA
    cudaEventRecord(stop_par, 0);                     // Запись времени окончания вычислений на GPU

    cudaMemcpy(&result_h, result_d, sizeof(double), cudaMemcpyDeviceToHost); // Копирование результата с GPU на CPU
    result_h *= h;                                    // Умножение на шаг интегрирования для получения итогового результата
    cudaEventSynchronize(stop_par);                   // Синхронизация событий

    float time_par;                                   // Переменная для хранения времени выполнения на GPU
    cudaEventElapsedTime(&time_par, start_par, stop_par); // Вычисление времени выполнения ядра CUDA

    printf("Результат параллельного вычисления: %.10f\n", result_h); // Вывод результата GPU
    printf("Время параллельного вычисления: %f миллисекунд\n", time_par); // Вывод времени GPU
    double acceleration = time_seq / time_par;        // Вычисление ускорения

    printf("Ускорение: %.2f раз\n", time_seq / time_par); // Вывод коэффициента ускорения

    // Проверка ошибок CUDA
    cudaError_t error = cudaGetLastError();           // Получение последней ошибки CUDA
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error)); // Вывод сообщения об ошибке
    }

    // Освобождение ресурсов
    cudaFree(result_d);                               // Освобождение памяти на устройстве
    cudaEventDestroy(start_par);                      // Удаление события начала
    cudaEventDestroy(stop_par);                       // Удаление события окончания
    return 0;                                         // Завершение программы
}
#pragma endregion