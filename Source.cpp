#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string>
#include<vector>
#include<iostream>
#include<random>

using namespace std;

class Data_Package {
private:
	int data_type;
	int output_num;
	int sequence_elements;

public:
	int data_size;
	int input_num;
	vector<vector<double>>load_data;
	vector<vector<double>>input_data;
	vector<vector<double>> ans_data;
	Data_Package(int data_size, int data_type, int input_num, int output_num, int sequence_elements) :
		data_size(data_size), data_type(data_type), input_num(input_num), output_num(output_num), sequence_elements(sequence_elements) {
		load_data.resize(this->data_size);
		for (int i = 0; i < this->data_size; i++) load_data[i].resize(this->data_type);
		input_data.resize(this->data_size);
		for (int i = 0; i < this->data_size; i++)input_data[i].resize(this->input_num);
		ans_data.resize(this->data_size);
		for (int i = 0; i < this->data_size; i++)ans_data[i].resize(this->output_num);

		FILE *in_file;
		in_file = fopen("DATA_One_Hot.dat", "r");
		int data_num = 0;
		while (fscanf(in_file, "%lf", &load_data[data_num][0]) != EOF && fscanf(in_file, "%lf", &load_data[data_num][1]) != EOF&&
			fscanf(in_file, "%lf", &load_data[data_num][2]) != EOF&&fscanf(in_file, "%lf", &load_data[data_num][3]) != EOF&&
			fscanf(in_file, "%lf", &load_data[data_num][4]) != EOF&&fscanf(in_file, "%lf", &load_data[data_num][5]) != EOF&&
			fscanf(in_file, "%lf", &load_data[data_num][6]) != EOF) {
			data_num++;
			if (data_num >= data_size)break;
		}
		fclose(in_file);
		double pos_amp = 100;
		for (int i = 0; i < this->data_size - this->sequence_elements; i++) {
			for (int j = 0; j < this->input_num; j++) {
				if (j % 2 == 0) input_data[i][j] = pos_amp*load_data[i + (int)(j / 2)][1];
				else input_data[i][j] = load_data[i + (int)(j / 2)][2];
			}
		}
		for (int i = this->data_size - this->sequence_elements; i < this->data_size; i++) {
			for (int j = 0; j < this->input_num; j++) {
				input_data[i][j] = 0;
			}
		}
		for (int i = 0; i < this->data_size - this->sequence_elements; i++) {
			ans_data[i][0] = (load_data[i + this->sequence_elements][1] - load_data[i][1]) / (load_data[i + this->sequence_elements][0] - load_data[i][0]);
			ans_data[i][1] = load_data[i + this->sequence_elements][2];
		}
		for (int i = this->data_size - this->sequence_elements; i < this->data_size; i++) {
			ans_data[i][0] = 0;
			ans_data[i][1] = 0;
		}
	}

	void Show_Data() {
		for (int i = 0; i < data_size; i++) {
			for (int j = 0; j < data_type; j++) {
				printf("%f ", load_data[i][j]);
			}
			cout << endl;
		}
		for (int i = 0; i < data_size; i++) {
			for (int j = 0; j < input_num; j++) {
				printf("%f ", input_data[i][j]);
			}
			cout << endl << endl;
		}
	}

	void Show_Ans() {
		for (int i = 0; i < data_size; i++) {
			printf("%f %f", ans_data[i][0], ans_data[i][1]);
			cout << endl;
		}
	}
};

class network_layer {
public:
	string act_func;
	int in_num;
	int out_num;
	vector<vector<double>> W;
	vector<double> b;
	vector<double> out_put;
	vector<double> delta;

	network_layer(string act_func, int in_num, int out_num) :
		act_func(act_func), in_num(in_num), out_num(out_num) {
		W.resize(this->out_num);
		for (int i = 0; i < this->out_num; i++) {
			W[i].resize(this->in_num);
		}
		b.resize(this->out_num);
		out_put.resize(this->out_num);
		delta.resize(this->out_num);

		double limit = sqrt(6 / (double)(this->in_num + this->out_num));
		random_device rnd;
		mt19937 mt(rnd());
		uniform_real_distribution<> randlimit(-limit, limit);
		for (int i = 0; i < this->out_num; i++) {
			b[i] = randlimit(mt);
			for (int j = 0; j < this->in_num; j++) {
				W[i][j] = randlimit(mt);
			}
		}
	}

	void show_Wb() {
		for (int i = 0; i < out_num; i++) {
			for (int j = 0; j < in_num; j++) printf("%f ", W[i][j]);
			cout << endl;
		}
		cout << endl;
		for (int i = 0; i < out_num; i++) printf("%f ", b[i]);
		cout << endl << endl;
	}
};

void dotWbh(network_layer *layer, vector<vector<double>>input_data, int data_index) {
	for (int i = 0; i < layer->out_num; i++) {
		layer->out_put[i] = layer->b[i];
		for (int j = 0; j < layer->in_num; j++) {
			layer->out_put[i] += layer->W[i][j] * input_data[data_index][j];
		}
	}
}
void dotWbh(network_layer *layer, network_layer *prev) {
	for (int i = 0; i < layer->out_num; i++) {
		layer->out_put[i] = layer->b[i];
		for (int j = 0; j < prev->out_num; j++) {
			layer->out_put[i] += layer->W[i][j] * prev->out_put[j];
		}
	}
}

void sigmoid_dotWbh(network_layer *layer, vector<vector<double>>input_data, int data_index) {
	for (int i = 0; i < layer->out_num; i++) {
		layer->out_put[i] = layer->b[i];
		for (int j = 0; j < layer->in_num; j++) {
			layer->out_put[i] += layer->W[i][j] * input_data[data_index][j];
		}
		layer->out_put[i] = 1 / (1 + exp(-layer->out_put[i]));
	}
}
void sigmoid_dotWbh(network_layer *layer, network_layer *prev) {
	for (int i = 0; i < layer->out_num; i++) {
		layer->out_put[i] = layer->b[i];
		for (int j = 0; j < layer->in_num; j++) {
			layer->out_put[i] += layer->W[i][j] * prev->out_put[j];
		}
		layer->out_put[i] = 1 / (1 + exp(-layer->out_put[i]));
	}
}

void tanh_dotWbh(network_layer *layer, vector<vector<double>>input_data, int data_index) {
	for (int i = 0; i < layer->out_num; i++) {
		layer->out_put[i] = layer->b[i];
		for (int j = 0; j < layer->in_num; j++) {
			layer->out_put[i] += layer->W[i][j] * input_data[data_index][j];
		}
		layer->out_put[i] = tanh(layer->out_put[i]);
	}
}
void tanh_dotWbh(network_layer *layer, network_layer *prev) {
	for (int i = 0; i < layer->out_num; i++) {
		layer->out_put[i] = layer->b[i];
		for (int j = 0; j < layer->in_num; j++) {
			layer->out_put[i] += layer->W[i][j] * prev->out_put[j];
		}
		layer->out_put[i] = tanh(layer->out_put[i]);
	}
}

void relu_dotWbh(network_layer *layer, vector<vector<double>>input_data, int data_index) {
	for (int i = 0; i < layer->out_num; i++) {
		layer->out_put[i] = layer->b[i];
		for (int j = 0; j < layer->in_num; j++) {
			layer->out_put[i] += layer->W[i][j] * input_data[data_index][j];
		}
		if (layer->out_put[i] <= 0)layer->out_put[i] = 0;
	}
}
void relu_dotWbh(network_layer *layer, network_layer *prev) {
	for (int i = 0; i < layer->out_num; i++) {
		layer->out_put[i] = layer->b[i];
		for (int j = 0; j < layer->in_num; j++) {
			layer->out_put[i] += layer->W[i][j] * prev->out_put[j];
		}
		if (layer->out_put[i] <= 0)layer->out_put[i] = 0;
	}
}

class AutoEncoder {
private:
	Data_Package *Data;
	int enc_layers_num;
	int dec_layers_num;
	double train_ratio;
	double dif_func;
	string loss_type;
	vector<double> y;
	vector<double>t;
	vector<double>E_y;
	vector<vector<double>>sum_end_delta;
	vector<vector<double>>sum_dec_delta;

public:
	network_layer **enc_layers;
	network_layer **dec_layers;
	AutoEncoder::AutoEncoder(Data_Package *Data, int enc_layers_num, network_layer **enc_layers, int dec_layers_num, network_layer **dec_layers, double train_ratio, string loss_type)
		:Data(Data), enc_layers_num(enc_layers_num), enc_layers(enc_layers), dec_layers_num(dec_layers_num), dec_layers(dec_layers), train_ratio(train_ratio), loss_type(loss_type) {
		y.resize(this->Data->input_num);
		t.resize(this->Data->input_num);
		E_y.resize(this->Data->input_num);
	}

	void encode(int data_index) {
		for (int i = 0; i < enc_layers_num; i++) {
			if (i == 0) dotWbh(enc_layers[i], Data->input_data, data_index);
			else dotWbh(enc_layers[i], enc_layers[i - 1]);
		}
	}
	void decode() {
		for (int i = 0; i < dec_layers_num; i++) {
			if (i == 0)dotWbh(dec_layers[i], enc_layers[enc_layers_num - 1]);
			else dotWbh(dec_layers[i], dec_layers[i - 1]);
		}
	}
	void enc_dec(int data_index) {
		for (int i = 0; i < enc_layers_num; i++) {
			if (enc_layers[i]->act_func == "linear") {
				if (i == 0) dotWbh(enc_layers[i], Data->input_data, data_index);
				else dotWbh(enc_layers[i], enc_layers[i - 1]);
			}
			else if (enc_layers[i]->act_func == "sigmoid") {
				if (i == 0) sigmoid_dotWbh(enc_layers[i], Data->input_data, data_index);
				else sigmoid_dotWbh(enc_layers[i], enc_layers[i - 1]);
			}
			else if (enc_layers[i]->act_func == "tanh") {
				if (i == 0) tanh_dotWbh(enc_layers[i], Data->input_data, data_index);
				else tanh_dotWbh(enc_layers[i], enc_layers[i - 1]);
			}
			else if (enc_layers[i]->act_func == "relu") {
				if (i == 0) relu_dotWbh(enc_layers[i], Data->input_data, data_index);
				else relu_dotWbh(enc_layers[i], enc_layers[i - 1]);
			}
		}
		for (int i = 0; i < dec_layers_num; i++) {
			if (dec_layers[i]->act_func == "linear") {
				if (i == 0)dotWbh(dec_layers[i], enc_layers[enc_layers_num - 1]);
				else dotWbh(dec_layers[i], dec_layers[i - 1]);
			}
			else if (dec_layers[i]->act_func == "sigmoid") {
				if (i == 0)sigmoid_dotWbh(dec_layers[i], enc_layers[enc_layers_num - 1]);
				else sigmoid_dotWbh(dec_layers[i], dec_layers[i - 1]);
			}
			else if (dec_layers[i]->act_func == "tanh") {
				if (i == 0)tanh_dotWbh(dec_layers[i], enc_layers[enc_layers_num - 1]);
				else tanh_dotWbh(dec_layers[i], dec_layers[i - 1]);
			}
			else if (dec_layers[i]->act_func == "relu") {
				if (i == 0)relu_dotWbh(dec_layers[i], enc_layers[enc_layers_num - 1]);
				else relu_dotWbh(dec_layers[i], dec_layers[i - 1]);
			}
		}
	}
	double calc_loss(int data_index) {
		double loss = 0;
		for (int i = 0; i < Data->input_num; i++) {
			if (loss_type == "mse") loss += pow(dec_layers[dec_layers_num - 1]->out_put[i] - Data->input_data[data_index][i], 2) / 2;
		}
		return loss;
	}
	void updata_parameters(double loss, double train_ratio, int data_index) {

		y = dec_layers[dec_layers_num - 1]->out_put;
		t = Data->input_data[data_index];
		for (int i = 0; i < Data->input_num; i++) {
			if (loss_type == "mse") E_y[i] = y[i] - t[i];
		}

		for (int i = dec_layers_num - 1; i >= 0; i--) {
			for (int j = 0; j < dec_layers[i]->in_num; j++) {
				for (int k = 0; k < dec_layers[i]->out_num; k++) {
					dec_layers[i]->delta[k] = 0;
					if (i == dec_layers_num - 1) {
						if (dec_layers[i]->act_func == "linear") dec_layers[i]->delta[k] = E_y[k];
						else if (dec_layers[i]->act_func == "sigmoid")dec_layers[i]->delta[k] = (1 - y[k])*y[k] * E_y[k];
						else if (dec_layers[i]->act_func == "tanh") dec_layers[i]->delta[k] = (1 - pow(y[k], 2))* E_y[k];
						else if (dec_layers[i]->act_func == "relu") {
							if (y[k] > 0)dec_layers[i]->delta[k] = E_y[k];
							else dec_layers[i]->delta[k] = 0;
						}
					}
					else {
						if (dec_layers[i]->act_func == "linear")for (int l = 0; l < dec_layers[i + 1]->out_num; l++) dec_layers[i]->delta[k] += dec_layers[i + 1]->delta[l] * dec_layers[i + 1]->W[l][k];
						else if (dec_layers[i]->act_func == "sigmoid")for (int l = 0; l < dec_layers[i + 1]->out_num; l++) dec_layers[i]->delta[k] += (1 - dec_layers[i]->out_put[k])*dec_layers[i]->out_put[k] * dec_layers[i + 1]->delta[l] * dec_layers[i + 1]->W[l][k];
						else if (dec_layers[i]->act_func == "tanh")for (int l = 0; l < dec_layers[i + 1]->out_num; l++) {
							dec_layers[i]->delta[k] += (1 - pow(dec_layers[i]->out_put[k], 2))* dec_layers[i + 1]->delta[l] * dec_layers[i + 1]->W[l][k];
						}
						else if (dec_layers[i]->act_func == "relu") {
							for (int l = 0; l < dec_layers[i + 1]->out_num; l++) {
								if (dec_layers[i]->out_put[k] > 0)dec_layers[i]->delta[k] += dec_layers[i + 1]->delta[l] * dec_layers[i + 1]->W[l][k];
							}
						}
					}
					if (i == 0) dec_layers[i]->W[k][j] -= train_ratio*dec_layers[i]->delta[k] * enc_layers[enc_layers_num - 1]->out_put[j];
					else dec_layers[i]->W[k][j] -= train_ratio*dec_layers[i]->delta[k] * dec_layers[i - 1]->out_put[j];
				}
			}
		}

		for (int i = enc_layers_num - 1; i >= 0; i--) {
			for (int j = 0; j < enc_layers[i]->in_num; j++) {
				for (int k = 0; k < enc_layers[i]->out_num; k++) {
					enc_layers[i]->delta[k] = 0;
					if (i == enc_layers_num - 1) {
						if (enc_layers[i]->act_func == "linear") for (int l = 0; l < dec_layers[0]->out_num; l++) enc_layers[i]->delta[k] += dec_layers[0]->delta[l] * dec_layers[0]->W[l][k];
						else if (enc_layers[i]->act_func == "sigmoid") for (int l = 0; l < dec_layers[0]->out_num; l++) enc_layers[i]->delta[k] += (1 - dec_layers[0]->out_put[k])*dec_layers[0]->out_put[k] * dec_layers[0]->delta[l] * dec_layers[0]->W[l][k];
						else if (enc_layers[i]->act_func == "tanh") for (int l = 0; l < dec_layers[0]->out_num; l++) enc_layers[i]->delta[k] += (1 - pow(dec_layers[0]->out_put[k], 2))* dec_layers[0]->delta[l] * dec_layers[0]->W[l][k];
						else if (enc_layers[i]->act_func == "relu") {
							for (int l = 0; l < dec_layers[0]->out_num; l++) {
								if (dec_layers[i]->out_put[k] > 0)enc_layers[i]->delta[k] += dec_layers[0]->delta[l] * dec_layers[0]->W[l][k];
							}
						}
					}
					else {
						if (enc_layers[i]->act_func == "linear")for (int l = 0; l < enc_layers[i + 1]->out_num; l++) enc_layers[i]->delta[k] += enc_layers[i + 1]->delta[l] * enc_layers[i + 1]->W[l][k];
						else if (enc_layers[i]->act_func == "sigmoid")for (int l = 0; l < enc_layers[i + 1]->out_num; l++) enc_layers[i]->delta[k] += (1 - enc_layers[i]->out_put[k])*enc_layers[i]->out_put[k] * enc_layers[i + 1]->delta[l] * enc_layers[i + 1]->W[l][k];
						else if (enc_layers[i]->act_func == "tanh")for (int l = 0; l < enc_layers[i + 1]->out_num; l++)enc_layers[i]->delta[k] += (1 - pow(enc_layers[i]->out_put[k], 2))* enc_layers[i + 1]->delta[l] * enc_layers[i + 1]->W[l][k];
						else if (enc_layers[i]->act_func == "relu") {
							for (int l = 0; l < enc_layers[i + 1]->out_num; l++) {
								if (enc_layers[i]->out_put[k] > 0)enc_layers[i]->delta[k] += enc_layers[i + 1]->delta[l] * enc_layers[i + 1]->W[l][k];
							}
						}
					}
					if (i == 0) enc_layers[i]->W[k][j] -= train_ratio*enc_layers[i]->delta[k] * Data->input_data[data_index][j];
					else enc_layers[i]->W[k][j] -= train_ratio*enc_layers[i]->delta[k] * enc_layers[i - 1]->out_put[j];
				}
			}
		}

	}
	void pretuning() {
		for (int i = 0; i < Data->data_size; i++) {
			enc_dec(i);
			updata_parameters(calc_loss(i), train_ratio, i);
		}
	}
	void finetuning() {}

	void print_Wb() {
		for (int i = 0; i < enc_layers_num; i++) {
			cout << "encoder layer" << i << " Weight" << endl;
			for (int j = 0; j < enc_layers[i]->out_num; j++) {
				for (int k = 0; k < enc_layers[i]->in_num; k++) printf("%f ", enc_layers[i]->W[j][k]);
				cout << endl;
			}
			cout << endl;
			cout << "encoder layer" << i << " bias" << endl;
			for (int j = 0; j < enc_layers[i]->out_num; j++) printf("%f ", enc_layers[i]->b[j]);
			cout << endl << endl;
		}
		for (int i = 0; i < dec_layers_num; i++) {
			cout << "decoder layer" << i << " Weight" << endl;
			for (int j = 0; j < dec_layers[i]->out_num; j++) {
				for (int k = 0; k < dec_layers[i]->in_num; k++) printf("%f ", dec_layers[i]->W[j][k]);
				cout << endl;
			}
			cout << endl;
			cout << "decoder layer" << i << " bias" << endl;
			for (int j = 0; j < dec_layers[i]->out_num; j++) printf("%f ", dec_layers[i]->b[j]);
			cout << endl << endl;
		}
	}
	void print_output(int data_index) {
		for (int i = 0; i < Data->input_num; i++) {
			printf("%d %f  %f ", i, dec_layers[dec_layers_num - 1]->out_put[i], Data->input_data[data_index][i]);
		}
		printf("\n");
	}
	void print_loss(int data_index) {
		double loss = 0;
		for (int i = 0; i < Data->input_num; i++) {
			loss += pow(dec_layers[dec_layers_num - 1]->out_put[i] - Data->input_data[data_index][i], 2) / 2;
		}
		printf("loss = %f\n", loss);
	}
};

int main() {
	const int sequence_elements = 30;
	const int elements_num = 2;
	const int input_num = sequence_elements*elements_num;
	const int output_num = 2;

	const int enc_layers_num = 3;
	string enc_act_func[enc_layers_num] = { "linear","linear","linear" };
	int enc_nodes[enc_layers_num + 1] = { input_num,30,10,3 };
	const int dec_layers_num = 3;
	string dec_act_func[dec_layers_num] = { "linear","linear","linear" };
	int dec_nodes[dec_layers_num + 1] = { 3,10,30,input_num };

	network_layer **enc_layers = new network_layer*[enc_layers_num];
	network_layer **dec_layers = new network_layer*[dec_layers_num];

	for (int i = 0; i < enc_layers_num; i++) enc_layers[i] = new network_layer(enc_act_func[i], enc_nodes[i], enc_nodes[i + 1]);
	for (int i = 0; i < dec_layers_num; i++) dec_layers[i] = new network_layer(dec_act_func[i], dec_nodes[i], dec_nodes[i + 1]);

	const int data_size = 1500;
	const int data_type = 7;

	Data_Package *Data;
	Data = new Data_Package(data_size, data_type, input_num, output_num, sequence_elements);

	double train_ratio = 0.0001;
	string loss_type = "mse";
	AutoEncoder *AE;
	AE = new AutoEncoder(Data, enc_layers_num, enc_layers, dec_layers_num, dec_layers, train_ratio, loss_type);

	//AE.print_Wb();
	//Data->Show_Data();
	//Data->Show_Ans();
	/*
	for (int j = 0; j < 10; j++) {
		for (int i = 0; i < 1000; i++) {
			AE.enc_dec(i);
			//AE.print_output();
			if (i % 100 == 0)AE.print_loss(i);
			AE.updata_parameters(AE.calc_loss(i), train_ratio, i);
		}
	}
	*/

	int data_index = 0;

	for (int i = 0; i < 10; i++) {
		printf("%d\n", i);
		AE->enc_dec(data_index);
		//AE->print_output(data_index);
		AE->print_loss(data_index);
		AE->updata_parameters(AE->calc_loss(data_index), train_ratio, data_index);
	}
	
	for (int i = 0; i < enc_layers_num; i++) {
		delete enc_layers[i];
	}
	for (int i = 0; i < dec_layers_num; i++) {
		delete dec_layers[i];
	}
	delete[] enc_layers;
	delete[] dec_layers;
	delete Data;
	delete AE;
	return 0;
}