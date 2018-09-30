class Neural {
private:
	Data_Package *Data;//�f�[�^
	int layers_num;//�G���R�[�_�w�̐�
	double train_ratio;//�w�K��
	std::string loss_type;//�����֐��̎��
	char loss_type_char;
	int batch_size;
	int batch_sample_num;
public:
	network_layer **layers;
	double *y;//�l�b�g���[�N�o�͒l
	double *t;//���t�f�[�^
	double *E_y;//�덷
	double ***sum_delta_W;
	double **sum_delta_b;
	double ***W_h;
	double **b_h;
	int **batch_index;
	int *shuffle;

	Neural(Data_Package *Data, int layers_num, network_layer **layers, double train_ratio=0, std::string loss_type="", int batch_size=0, int batch_sample_num=0);
	~Neural();

	void set_train_batch();
	/*�v�Z�p�֐�*/
	void act_func_dotWb(network_layer *layer, double ** input_data, int data_index);
	void act_func_dotWb(network_layer *layer, network_layer *prev);

	/*�G���R�[�h*/
	void encode(int data_index);

	/*�����֐�*/
	double calc_loss(int data_index);

	/*�p�����[�^�X�V*/
	void updata_parameters(double train_ratio, int batch_size);

	/*���O�w�K*/
	void pretuning();

	/*���t����w�K*/
	void finetuning();

	/*�`��p�֐�*/
	void print_Wb();
	double print_output(int data_index);
	void print_loss(int batch_size);
	void print_index_loss(int data_index);
	void plot_graph();

	/*�p�����[�^�̕ۑ��E���[�h*/
	void save_Wb();
	void load_Wb();
};