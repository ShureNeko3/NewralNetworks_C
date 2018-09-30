class Neural {
private:
	Data_Package *Data;//データ
	int layers_num;//エンコーダ層の数
	double train_ratio;//学習率
	std::string loss_type;//損失関数の種類
	char loss_type_char;
	int batch_size;
	int batch_sample_num;
public:
	network_layer **layers;
	double *y;//ネットワーク出力値
	double *t;//教師データ
	double *E_y;//誤差
	double ***sum_delta_W;
	double **sum_delta_b;
	double ***W_h;
	double **b_h;
	int **batch_index;
	int *shuffle;

	Neural(Data_Package *Data, int layers_num, network_layer **layers, double train_ratio=0, std::string loss_type="", int batch_size=0, int batch_sample_num=0);
	~Neural();

	void set_train_batch();
	/*計算用関数*/
	void act_func_dotWb(network_layer *layer, double ** input_data, int data_index);
	void act_func_dotWb(network_layer *layer, network_layer *prev);

	/*エンコード*/
	void encode(int data_index);

	/*損失関数*/
	double calc_loss(int data_index);

	/*パラメータ更新*/
	void updata_parameters(double train_ratio, int batch_size);

	/*事前学習*/
	void pretuning();

	/*教師あり学習*/
	void finetuning();

	/*描画用関数*/
	void print_Wb();
	double print_output(int data_index);
	void print_loss(int batch_size);
	void print_index_loss(int data_index);
	void plot_graph();

	/*パラメータの保存・ロード*/
	void save_Wb();
	void load_Wb();
};