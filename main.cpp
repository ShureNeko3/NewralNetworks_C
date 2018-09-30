#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string>
#include<iostream>
#include<random>
#include<time.h>
#include"utils.h"
#include"network.h"
#include"neural.h"

void mydebug(int line) {
	printf("no problem until this line %d\n", line);
}

int main() {
	//*************************//
	//**  ユーザ設定変数一覧  **//
	//************************//

	///////////////////////////
	/*データについての固定変数*/
	//////////////////////////
	FILE *file_info;
	file_info = fopen("../data_size.dat", "r");
	//ファイルの数
	int file_size;
	//データの種類（時間，温度，電力，etc）
	int file_num;
	fscanf(file_info, "%d %d", &file_size, &file_num);
	fclose(file_info);

	//////////////////////////////
	/*データについての任意設定変数*/
	//////////////////////////////
	//時系列データの時間幅
	int sequence_elements = 1;
	//入力データの種類
	int input_elements_num = 1;
	int *input_index;
	input_index = new int[input_elements_num];
	//出力データの種類
	int output_elements_num = 1;
	//回帰量を含めた入力データの数
	int input_num = input_elements_num * sequence_elements + output_elements_num*(sequence_elements - 1);
	//y[k]=f(y[k-1]...,y[k-sequence_elements+1],x[k],x[k-1],...,x[k-sequence_elements+1])
	int output_num = output_elements_num;
	//一つのデータから生成されるデータサイズ
	int data_size = file_size - sequence_elements + 1;

	///////////////////////////
	/*NNの層についての設定変数*/
	//////////////////////////
	//NNの層の数(任意)
	const int layers_num = 3;
	//NNのノードの数（任意）
	int nodes[layers_num + 1] = { input_num,20,20,output_num };
	//活性化関数("linear"or"sigmoid"or"tanh"or"relu")
	std::string act_func[layers_num] = { "sigmoid","sigmoid","linear" };

	/////////////////////////
	/*学習についての設定変数*/
	////////////////////////
	//学習率(現在は"AdaGrad")
	double train_ratio = 0.010;
	//損失関数("mse"(=mean square error))
	std::string loss_type = "mse";
	//学習回数
	int epoch = 10000;
	int print_epoch = 100;
	//バッチサイズ
	int batch_size = data_size;
	int batch_sample_num = data_size / batch_size;
	
	///////////////////////
	/*オブジェクトの初期化*/
	//////////////////////
	Data_Package *Data;
	char data_name[] = "../データ100体分 - 提出用.csv";
	//file_dataからinput_dataを選択する(input_elements_num個選ぶこと(0～file_num-1))
	input_index[0] = 14;//1,5,15,16
	//input_index[1] = 14;
	/*
	input_index[1] = 3;
	input_index[2] = 5;
	input_index[3] = 7;
	input_index[4] = 5;
	input_index[5] = 6;
	input_index[6] = 7;
	input_index[7] = 8;
	*/
	Data = new Data_Package(data_name, sequence_elements, input_elements_num, input_num, output_num, file_size, data_size, file_num,input_index);
	Data->Show_Data();
	
	network_layer **layers = new network_layer*[layers_num];
	for (int i = 0; i < layers_num; i++) layers[i] = new network_layer(act_func[i], nodes[i], nodes[i + 1]);

	Neural *NN;
	NN = new Neural(Data, layers_num, layers, train_ratio, loss_type, batch_size, batch_sample_num);
	NN->load_Wb();
	
	//********************//
	//**  学習フェーズ  **//
	//*******************//
	
	clock_t start = clock();
	for (int i = 0; i < epoch; i++) {
		NN->set_train_batch();
		NN->updata_parameters(train_ratio, batch_size);
		if (i%print_epoch == 0) {
			printf("epoch = %d ", i + 1);
			NN->print_loss(batch_size);
		}
	}
	clock_t end = clock();
	std::cout << "learning time = " << (double)(end - start) / CLOCKS_PER_SEC << "sec" << std::endl;
	
	//******************//
	//**  結果の出力  **//
	//*****************//
	NN->plot_graph();
	NN->save_Wb();
	double predict_error;
	double ave_error = 0.0;
	FILE *fp_error;
	fp_error = fopen("./data/error.dat", "w");
	for (int i = 0; i < data_size; i++) {
		predict_error=NN->print_output(i);
		fprintf(fp_error,"%d %lf\n", i,predict_error);
		ave_error += abs(predict_error);
	}
	fclose(fp_error);
	printf("average error = %lf\n", (double)ave_error / data_size);

	///////////////////////
	/*オブジェクトの後処理*/
	//////////////////////
	delete[] input_index;
	delete NN;
	for (int i = 0; i < layers_num; i++) delete layers[i];
	delete[] layers;
	delete Data;
}