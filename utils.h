class Data_Package {
public:
	int file_num;//データの種類
	int data_size;
	int input_num;//入力層のノード数
	int output_num;//出力データの種類
	int sequence_elements;//時系列データの時間幅
	int file_size;
	Data_Package(char data_name[], int sequence_elements, int input_elements_num, int input_num, int output_num, int file_size, int data_size, int file_num,int *input_index);
	~Data_Package(void);
	void Show_Data();
	void Show_Ans();
	double **load_data;//生データ
	double **input_data;//入力用に加工したデータ
	double **ans_data;//出力用(教師用)に加工したデータ
	double *input_max_value;
};