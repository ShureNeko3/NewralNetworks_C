class Data_Package {
public:
	int file_num;//�f�[�^�̎��
	int data_size;
	int input_num;//���͑w�̃m�[�h��
	int output_num;//�o�̓f�[�^�̎��
	int sequence_elements;//���n��f�[�^�̎��ԕ�
	int file_size;
	Data_Package(char data_name[], int sequence_elements, int input_elements_num, int input_num, int output_num, int file_size, int data_size, int file_num,int *input_index);
	~Data_Package(void);
	void Show_Data();
	void Show_Ans();
	double **load_data;//���f�[�^
	double **input_data;//���͗p�ɉ��H�����f�[�^
	double **ans_data;//�o�͗p(���t�p)�ɉ��H�����f�[�^
	double *input_max_value;
};