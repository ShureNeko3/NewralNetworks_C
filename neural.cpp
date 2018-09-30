#include<stdio.h>
#include<string>
#include<random>
#include"utils.h"
#include"network.h"
#include"neural.h"

Neural::Neural(Data_Package *Data, int layers_num, network_layer **layers, double train_ratio, std::string loss_type, int batch_size, int batch_sample_num)
	:Data(Data), layers_num(layers_num), layers(layers), train_ratio(train_ratio), loss_type(loss_type), batch_size(batch_size), batch_sample_num(batch_sample_num) {
	y = new double[Data->output_num];
	t = new double[Data->output_num];
	E_y = new double[Data->output_num];
	sum_delta_W = new double**[layers_num];
	W_h = new double**[layers_num];
	for (int i = 0; i < layers_num; i++) {
		sum_delta_W[i] = new double*[layers[i]->out_num];
		W_h[i] = new double*[layers[i]->out_num];
		for (int j = 0; j < layers[i]->out_num; j++) {
			sum_delta_W[i][j] = new double[layers[i]->in_num];
			W_h[i][j] = new double[layers[i]->in_num];
		}
	}
	sum_delta_b = new double*[layers_num];
	b_h = new double*[layers_num];
	for (int i = 0; i < this->layers_num; i++) {
		sum_delta_b[i] = new double[layers[i]->out_num];
		b_h[i] = new double[layers[i]->out_num];
	}
	batch_index = new int*[this->batch_sample_num];
	for (int i = 0; i < this->batch_sample_num; i++)batch_index[i] = new int[this->batch_size];
	shuffle = new int[Data->data_size];
	loss_type_char = loss_type[0];
	for (int i = 0; i < layers_num; i++) {
		for (int j = 0; j < layers[i]->out_num; j++) {
			b_h[i][j] = 0.000000010;
			for (int k = 0; k < layers[i]->in_num; k++) {
				W_h[i][j][k] = 0.000000010;
			}
		}
	}
}
Neural::~Neural() {
	delete[] E_y;
	for (int i = 0; i < layers_num; i++) {
		for (int j = 0; j < layers[i]->out_num; j++) {
			delete[] sum_delta_W[i][j];
			delete[] W_h[i][j];
		}
		delete[] sum_delta_W[i];
		delete[] W_h[i];
	}
	delete[] sum_delta_W;
	delete[] W_h;
	for (int i = 0; i < layers_num; i++) {
		delete[] sum_delta_b[i];
		delete[] b_h[i];
	}
	delete[] sum_delta_b;
	delete[] b_h;
	for (int i = 0; i < batch_sample_num; i++) {
		delete[] batch_index[i];
	}
	delete[] batch_index;
	delete[] shuffle;
}

void Neural::set_train_batch() {
	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_int_distribution<> rand(0, Data->data_size - 1);
	for (int i = 0; i < Data->data_size; i++) {
		shuffle[i] = i;
	}
	for (int i = 0; i < Data->data_size; i++) {
		int rnd = rand(mt);
		int temp = shuffle[i];
		shuffle[i] = shuffle[rnd];
		shuffle[rnd] = temp;
	}
	for (int i = 0; i < batch_sample_num; i++) {
		for (int j = 0; j < batch_size; j++) {
			batch_index[i][j] = shuffle[i*batch_size + j];
		}
	}
}

/*計算用関数*/
void Neural::act_func_dotWb(network_layer *layer, double ** input_data, int data_index) {
	for (int i = 0; i < layer->out_num; i++) {
		layer->out_put[i] = layer->b[i];
		for (int j = 0; j < layer->in_num; j++) {
			layer->out_put[i] += layer->W[i][j] * input_data[data_index][j];
		}
		if (layer->act_func_char == 'l');
		else if (layer->act_func_char == 's')layer->out_put[i] = 1 / (1 + exp(-layer->out_put[i]));
		else if (layer->act_func_char == 't')layer->out_put[i] = tanh(layer->out_put[i]);
		else if (layer->act_func_char == 'r'&&layer->out_put[i] <= 0)layer->out_put[i] = 0;
	}
}
void Neural::act_func_dotWb(network_layer *layer, network_layer *prev) {
	for (int i = 0; i < layer->out_num; i++) {
		layer->out_put[i] = layer->b[i];
		for (int j = 0; j < prev->out_num; j++) {
			layer->out_put[i] += layer->W[i][j] * prev->out_put[j];
		}
		if (layer->act_func_char == 'l');
		else if (layer->act_func_char == 's')layer->out_put[i] = 1 / (1 + exp(-layer->out_put[i]));
		else if (layer->act_func_char == 't')layer->out_put[i] = tanh(layer->out_put[i]);
		else if (layer->act_func_char == 'r'&&layer->out_put[i] <= 0)layer->out_put[i] = 0;
	}
}

/*エンコード*/
void Neural::encode(int data_index) {
	for (int i = 0; i < layers_num; i++) {
		if (i == 0) act_func_dotWb(layers[i], Data->input_data, data_index);
		else act_func_dotWb(layers[i], layers[i - 1]);
	}
}

/*損失関数*/
double Neural::calc_loss(int data_index) {
	double loss = 0;
	for (int i = 0; i < Data->output_num; i++) {
		if (loss_type == "mse") loss += pow(layers[layers_num - 1]->out_put[i] - Data->ans_data[data_index][i], 2) / 2;
	}
	return loss;
}

/*パラメータ更新*/
void Neural::updata_parameters(double train_ratio, int batch_size) {
	int data_index;
	for (int sample = 0; sample < batch_sample_num; sample++) {

		for (int i = layers_num - 1; i >= 0; i--) {
			for (int j = 0; j < layers[i]->in_num; j++) {
				for (int k = 0; k < layers[i]->out_num; k++) {
					sum_delta_W[i][k][j] = 0;
					if (j == 0)sum_delta_b[i][k] = 0;
				}
			}
		}

		for (int index = 0; index < batch_size; index++) {
			data_index = batch_index[sample][index];
			encode(data_index);
			y = layers[layers_num - 1]->out_put;
			t = Data->ans_data[data_index];
			for (int i = 0; i < Data->output_num; i++) {
				if (loss_type_char == 'm') E_y[i] = y[i] - t[i];
			}

			for (int i = layers_num - 1; i >= 0; i--) {
				for (int j = 0, jj = layers[i]->in_num; j < jj; j++) {
					for (int k = 0, kk = layers[i]->out_num; k < kk; k++) {
						layers[i]->delta[k] = 0;
						if (i != layers_num - 1) {
							if (layers[i]->act_func_char == 'l')for (int l = 0, ll = layers[i + 1]->out_num; l < ll; l++) layers[i]->delta[k] += layers[i + 1]->delta[l] * layers[i + 1]->W[l][k];
							else if (layers[i]->act_func_char == 's')for (int l = 0, ll = layers[i + 1]->out_num; l < ll; l++) layers[i]->delta[k] += (1 - layers[i]->out_put[k])*layers[i]->out_put[k] * layers[i + 1]->delta[l] * layers[i + 1]->W[l][k];
							else if (layers[i]->act_func_char == 't')for (int l = 0, ll = layers[i + 1]->out_num; l < ll; l++) {
								layers[i]->delta[k] += (1 - pow(layers[i]->out_put[k], 2))* layers[i + 1]->delta[l] * layers[i + 1]->W[l][k];
							}
							else if (layers[i]->act_func_char == 'r') {
								for (int l = 0, ll = layers[i + 1]->out_num; l < ll; l++) {
									if (layers[i]->out_put[k] > 0)layers[i]->delta[k] += layers[i + 1]->delta[l] * layers[i + 1]->W[l][k];
								}
							}
						}
						else {
							if (layers[i]->act_func_char == 'l') layers[i]->delta[k] = E_y[k];
							else if (layers[i]->act_func_char == 's')layers[i]->delta[k] = (1 - y[k])*y[k] * E_y[k];
							else if (layers[i]->act_func_char == 't') layers[i]->delta[k] = (1 - pow(y[k], 2))* E_y[k];
							else if (layers[i]->act_func_char == 'r') {
								if (y[k] > 0)layers[i]->delta[k] = E_y[k];
								else layers[i]->delta[k] = 0;
							}
						}
						if (i == 0) sum_delta_W[i][k][j] += layers[i]->delta[k] * Data->input_data[data_index][j];
						else sum_delta_W[i][k][j] += layers[i]->delta[k] * layers[i - 1]->out_put[j];
						if (j == 0) {
							if (i == 0) sum_delta_b[i][k] += layers[i]->delta[k];
							else sum_delta_b[i][k] += layers[i]->delta[k];
						}
					}
				}
			}
		}

		for (int i = layers_num - 1; i >= 0; i--) {
			for (int j = 0; j < layers[i]->in_num; j++) {
				for (int k = 0; k < layers[i]->out_num; k++) {
					W_h[i][k][j] += pow(sum_delta_W[i][k][j], 2);
					layers[i]->W[k][j] -= train_ratio*sum_delta_W[i][k][j] / sqrt(W_h[i][k][j]);
					if (j == 0) {
						b_h[i][k] += pow(sum_delta_b[i][k], 2);
						layers[i]->b[k] -= train_ratio*sum_delta_b[i][k] / sqrt(b_h[i][k]);
					}
				}
			}
		}
	}
}

/*事前学習*/
void Neural::pretuning() {
	for (int i = 0; i < Data->data_size; i++) {
		encode(i);
		//updata_parameters(calc_loss(i), train_ratio, i);
	}
}

/*教師あり学習*/
void Neural::finetuning() {

}

/*描画用関数*/
void Neural::print_Wb() {
	for (int i = 0; i < layers_num; i++) {
		printf("encoder layer %d Weight", i);
		for (int j = 0; j < layers[i]->out_num; j++) {
			for (int k = 0; k < layers[i]->in_num; k++) printf("%f ", layers[i]->W[j][k]);
			printf("\n");
		}
		printf("\n");
		printf("encoder layer %d bias", i);
		for (int j = 0; j < layers[i]->out_num; j++) printf("%f ", layers[i]->b[j]);
		printf("\n\n");
	}
}
double Neural::print_output(int data_index) {
	printf("data_index=%d\n", data_index);
	encode(data_index);
	double ave_error = 0.0;
	for (int i = 0; i < Data->output_num; i++) {
		printf("data = %f, output = %f, error = %f\n", Data->ans_data[data_index][i]*Data->input_max_value[0], layers[layers_num - 1]->out_put[i] * Data->input_max_value[0], layers[layers_num - 1]->out_put[i] * Data->input_max_value[0]-Data->ans_data[data_index][i] * Data->input_max_value[0]);
		ave_error=layers[layers_num - 1]->out_put[i] * Data->input_max_value[0] - Data->ans_data[data_index][i] * Data->input_max_value[0];
	}
	return ave_error;
}
void Neural::print_loss(int batch_size) {
	double loss = 0;
	int data_index;
	for (int sample = 0; sample < batch_sample_num; sample++) {
		for (int index = 0; index < batch_size; index++) {
			data_index = batch_index[sample][index];
			encode(data_index);
			for (int i = 0; i < Data->output_num; i++) loss += pow(layers[layers_num - 1]->out_put[i] - Data->ans_data[data_index][i], 2);
		}
	}
	printf("loss = %f\n", sqrt(loss) / (batch_size));
}
void Neural::print_index_loss(int data_index) {
	double loss = 0;
	for (int i = 0; i < Data->output_num; i++) if (loss_type == "mse") loss += pow(layers[layers_num - 1]->out_put[i] - Data->ans_data[data_index][i], 2) / 2;
	printf("%d %f\n", data_index, loss);
}
void Neural::plot_graph() {
	FILE *gp;
	FILE *fp,*fp2,*fp3;
	fp = fopen("./data/kona.dat", "w");
	fp2 = fopen("./data/kona_fail.dat", "w");
	fp3 = fopen("./data/predict.dat", "w");
	for (int i = 0; i < Data->data_size; i++) {
		encode(i);
		fprintf(fp, "%d %f %f\n", i,Data->load_data[i][0]*Data->input_max_value[0], layers[layers_num - 1]->out_put[0]*Data->input_max_value[0]);
		if (abs(Data->load_data[i][0] - layers[layers_num - 1]->out_put[0])*Data->input_max_value[0] > 1.0) {
			fprintf(fp2, "%d %f %f ", i, Data->load_data[i][0] * Data->input_max_value[0], layers[layers_num - 1]->out_put[0] * Data->input_max_value[0]);
			for (int j = 0; j < Data->file_num; j++)fprintf(fp2, "%lf ", Data->load_data[i][j]);
			fprintf(fp2,"\n");
			printf("%d %lf\n", i, (Data->load_data[i][0] - layers[layers_num - 1]->out_put[0])*Data->input_max_value[0]);
		}
		fprintf(fp3, "%f %f %f\n", Data->load_data[i][14], layers[layers_num - 1]->out_put[0],Data->ans_data[i][0]);
	}
	fclose(fp),fclose(fp2),fclose(fp3);
	gp = _popen("gnuplot -persist", "w");
	fprintf(gp, "set terminal postscript eps enhanced color\nset output './figure/kona.eps'\n");
	//fprintf(gp, "set xrange[132:140]\n");
	//fprintf(gp, "set yrange[132:140]\n");
	fprintf(gp, "set xlabel 'Value of data [g]'\n");
	fprintf(gp, "set ylabel 'Value of NN output [g]'\n");
	fprintf(gp, "set size square\n");
	fprintf(gp, "plot x title 'y = x',");
	fprintf(gp, "x+1 title 'y = x+1',");
	fprintf(gp, "x-1 title 'y = x-1',");
	fprintf(gp, "'./data/kona.dat' using 2:3 pt 1 lc 'red' title '',");
	fprintf(gp, "'./data/kona_fail.dat' using 2:3 pt 7 lc 'blue' title ''\n");
	fprintf(gp, "set terminal postscript eps enhanced color\nset output'./figure/predict.eps'\n");
	fprintf(gp, "set xlabel'tag=1'\n");
	fprintf(gp, "set ylabel'value of NN output'\n");
	fprintf(gp, "plot './data/predict.dat' using 1:2 pt 7 lc 'blue' title '',");
	fprintf(gp, "'./data/predict.dat' using 1:3 pt 7 lc 'red' title ''\n");
	fprintf(gp, "quit\n");
	fflush(gp);
	_pclose(gp);
}
void Neural::save_Wb() {
	FILE *fp1, *fp2;
	fp1 = fopen("../Weight.dat", "w");
	fp2 = fopen("../Bias.dat", "w");
	for (int i = 0; i < layers_num; i++) {
		for (int j = 0; j < layers[i]->out_num; j++) {
			for (int k = 0; k < layers[i]->in_num; k++) {
				fprintf(fp1, " %f", layers[i]->W[j][k]);
			}
			fprintf(fp1, "\r\n");
		}
	}
	for (int i = 0; i < layers_num; i++) {
		for (int j = 0; j < layers[i]->out_num; j++) {
			fprintf(fp2, " %f", layers[i]->b[j]);
		}
		fprintf(fp2, "\r\n");
	}
	fclose(fp1), fclose(fp2);
}
void Neural::load_Wb() {
	FILE *fp1, *fp2;
	fp1 = fopen("../Weight.dat", "r");
	fp2 = fopen("../Bias.dat", "r");
	for (int i = 0; i < layers_num; i++) {
		for (int j = 0; j < layers[i]->out_num; j++) {
			fscanf(fp2, "%lf", &layers[i]->b[j]);
			for (int k = 0; k < layers[i]->in_num; k++) {
				fscanf(fp1, "%lf", &layers[i]->W[j][k]);
			}
		}
	}
	fclose(fp1), fclose(fp2);
}