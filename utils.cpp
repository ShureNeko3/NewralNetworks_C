#include<stdio.h>
#include<string>
#include<math.h>
#include"utils.h"
Data_Package::Data_Package(char data_name[], int sequence_elements, int input_elements_num, int input_num, int output_num, int file_size, int data_size, int file_num, int *input_index) :
	sequence_elements(sequence_elements), data_size(data_size), input_num(input_num), output_num(output_num), file_size(file_size), file_num(file_num) {
	load_data = new double*[file_size];
	for (int i = 0; i < file_size; i++) load_data[i] = new double[file_num];
	input_data = new double*[data_size];
	for (int i = 0; i < data_size; i++)input_data[i] = new double[input_num];
	ans_data = new double*[data_size];
	for (int i = 0; i < data_size; i++)ans_data[i] = new double[output_num];
	input_max_value = new double[file_num];
	FILE *in_file;
	in_file = fopen(data_name, "r");
	char str[1024];
	char *p;
	for (int i = 0; i < file_num; i++)input_max_value[i] = 0;
	for (int i = 0; i < 7; i++) fgets(str, sizeof(str), in_file);
	for (int i = 0; i < file_size; i++) {
		fgets(str, sizeof(str), in_file);
		p = strtok(str, ",");
		p = strtok(NULL, ",");
		for (int j = 0; j < file_num; j++) {
			p = strtok(NULL, ",");
			load_data[i][j] = atof(p);
			if (abs(load_data[i][j]) > input_max_value[j])input_max_value[j] = abs(load_data[i][j]);
		}
	}
	for (int i = 0; i < file_size; i++) {
		for (int j = 0; j < file_num; j++) {
			load_data[i][j] /= input_max_value[j];
		}
	}
	fclose(in_file);
	for (int i = 0; i < data_size; i++) {
		for (int j = 0; j < sequence_elements; j++) {
			for (int k = 0; k < input_elements_num; k++) {
				input_data[i][j*input_elements_num + k] = load_data[i + j][input_index[k]];
				//printf("%d %d %d %lf \n", i, j, k, input_data[i][j*file_num + k]);
			}
			if (j == sequence_elements - 1)continue;
			input_data[i][sequence_elements*input_elements_num + j] = load_data[i + j][0];
		}
		ans_data[i][0] = load_data[i + sequence_elements - 1][0];
		//printf("%d %lf\n", i, ans_data[i][0]);
		//printf("\n");
	}
}
Data_Package::~Data_Package() {
	for (int i = 0; i < file_size; i++) delete[] load_data[i];
	for (int i = 0; i < data_size; i++) delete[] input_data[i];
	for (int i = 0; i < data_size; i++) delete[] ans_data[i];
	delete[] load_data;
	delete[] input_data;
	delete[] ans_data;
	delete[] input_max_value;
}
/*•`‰æ—pŠÖ”*/
void Data_Package::Show_Data() {
	for (int i = 0; i < file_size; i++) {
		printf("%d ", i);
		for (int j = 0; j < file_num; j++) {
			printf("%f ", load_data[i][j]);
		}
		printf("\n");
	}

	for (int i = 0; i < data_size; i++) {
		printf("%d ", i);
		for (int j = 0; j < input_num; j++) {
			printf("%f ", input_data[i][j]);
		}
		printf("\n");
	}

	for (int i = 0; i < data_size; i++) {
		printf("%d ", i);
		for (int j = 0; j < output_num; j++) {
			printf("%f ", ans_data[i][j]);
		}
		printf("\n");
	}
}
void Data_Package::Show_Ans() {
	for (int i = 0; i < data_size; i++) {
		for (int j = 0; j < output_num; j++) printf("%f", ans_data[i][j]);
		printf("\n");
	}
}