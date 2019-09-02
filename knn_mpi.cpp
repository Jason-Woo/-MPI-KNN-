#include<iostream>
#include<fstream>
#include<string>
#include<string.h>
#include<cmath>
#include<sstream>
#include"mpi.h"
#include<algorithm>
#include<time.h>

using namespace std;

struct train_data_dis
{
	/*
	train_data_dis结构体储存了测试集或验证集的某一个点与训练集中某一个点的距离以及这个点的类标。通过引入这个结构体我们可以更方便地在KNN的过程中选出距离较小的K个样本点以及对应类标。
	label：该训练集样本的类标
	dis：测试集或验证集与该训练集样本的距离
	*/
	int label;
	double dis;
};

bool Comp(train_data_dis a, train_data_dis b)
{
	/*
	Comp函数自定义了快速排序算法sort的比较函数。通过这种方式我们可以对train_data_dis类的对象根据储存的距离大小进行排序。
	a和b是需要比较的两个对象
	*/
	return a.dis < b.dis;
}

double Euclidean_D(int a, int b, int dim, double* Data_train, double* Data_test_buffer)
{
	/*
	Euclidean_D函数返回的是两个样本点之间的距离，计算使用的的方式是欧式距离。
	a：测试集或验证集目标样本点编号
	b：训练集目标样本点编号
	dim：样本点的维度
	Data_train：完整训练集
	Data_test_buffer：储存在该进程缓冲区内的测试集或验证集
	*/
	double result = 0;
	for (int i = 0; i < dim; i++)
	{
		result += (Data_test_buffer[a * dim + i] - Data_train[b * dim + i]) * (Data_test_buffer[a * dim + i] - Data_train[b * dim + i]);
	}
	result = sqrt(result);
	return result;
}
double Manhattan_D(int a, int b, int dim, double* Data_train, double* Data_test_buffer)
{
	/*
	Manhattan_D函数返回的是两个样本点之间的距离，计算使用的的方式是曼哈顿距离。
	a：测试集或验证集目标样本点编号
	b：训练集目标样本点编号
	dim：样本点的维度
	Data_train：完整训练集
	Data_test_buffer：储存在该进程缓冲区内的测试集或验证集
	*/
	double result = 0;
	for (int i = 0; i < dim; i++)
	{
		result += abs(Data_test_buffer[a * dim + i] - Data_train[b * dim + i]);
	}
	return result;
}

double acc_calc(int* real_label, int* label, int size)
{
	/*
	acc_calc是用于计算算法验证集预测结果的准确率的，输入验证集的真实类标以及KNN预测类标，返回值准确率。
	real_label：验证集的真实类标
	label：算法预测出的验证集类标
	size：验证集大小
	*/
	double acc = 0;
	for (int i = 0; i < size; i++)
	{
		if (real_label[i] == label[i]) acc++;
	}
	acc /= size;
	return acc;
}

int main(int argc, char* argv[])
{
	int dim, N_train, N_test, N_val, K, class_cnt;
	bool Euclidean_distance, Normalize, Validation;

	double* max_value, * min_value;
	double* max_value_buffer, * min_value_buffer;

	double* Data_train; //训练集样本
	int* Train_label; //训练集类标

	double* Data_test; //测试集样本
	double* Data_test_buffer; //该进程缓冲区内测试集样本
	int* Test_label; //测试集预测类标
	int* Test_label_buffer; //该进程缓冲区内测试集预测类标

	double* Data_val; //验证集样本
	double* Data_val_buffer; //该进程缓冲区内验证集样本
	int* Val_label; //验证集预测类标
	int* Val_label_buffer; //该进程缓冲区内验证集预测类标
	int* Val_label_real; //验证集真实类标

	dim = 784; //样本点的维度
	K = 50; //KNN中的K值
	N_train = 60000; //训练集大小
	N_test = 10000; //测试集大小
	N_val = 10000; //验证集大小
	class_cnt = 10; //总类标数
	Euclidean_distance = true; //是否适用欧式距离，为false则使用曼哈顿距离
	Normalize = true; //是否归一化，为false则不对训练、验证、测试集数据进行归一化
	Validation = true; //是否使用验证集，为false则不引入验证集
	string train_file_name = "mnist_train.csv"; //训练集的存储路径及文件名
	string validation_file_name = "mnist_validation.csv"; //验证集的存储路径及文件名
	string test_file_name = "mnist_test.csv"; //测试集的存储路径及文件名

	int myid, numprocs;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid); //获取当前进程id
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs); //获取总进程数

	if (N_train % numprocs != 0) MPI_Abort(MPI_COMM_WORLD, 1);
	if (N_test % numprocs != 0) MPI_Abort(MPI_COMM_WORLD, 1);
	if (N_val % numprocs != 0) MPI_Abort(MPI_COMM_WORLD, 1);

	double start, finish;
	double totaltime; //算法的总运行时间
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();

	int batch_test = N_test / numprocs;
	int batch_train = N_train / numprocs; 
	int batch_val = N_val / numprocs;

	Data_train = new double[N_train * dim];
	Train_label = new int[N_train];

	Data_test = new double[N_test * dim];
	Data_test_buffer = new double[batch_test * dim];
	Test_label = new int[N_test];
	Test_label_buffer = new int[batch_test];

	Data_val = new double[N_val * dim];
	Val_label_real = new int[N_val];
	Data_val_buffer = new double[batch_val * dim];
	Val_label = new int[N_val];
	Val_label_buffer = new int[batch_val];

	if (myid == 0)
	{
		/*
		进程0读取训练集并将属性信息和类标分别储存
		*/
		ifstream infile;
		infile.open(train_file_name);
		string s;
		int cnt = 0;
		while (getline(infile, s))
		{
			stringstream ss(s);
			string a;
			while (getline(ss, a, ','))
			{
				if (cnt % (dim + 1) == 0) Train_label[cnt / (dim + 1)] = atoi(a.c_str());
				else Data_train[cnt - (cnt / (dim + 1)) - 1] = atof(a.c_str());
				cnt++;
			}
		}
		infile.close();
	}

	if (myid == 1)
	{
		/*
		进程1读取测试集并储存属性信息
		*/
		ifstream infile;
		infile.open(test_file_name);
		string s;
		int cnt = 0;
		while (getline(infile, s))
		{
			stringstream ss(s);
			string a;
			while (getline(ss, a, ','))
			{
				Data_test[cnt] = atof(a.c_str());
				cnt++;
			}
		}
		infile.close();
	}
	if (Validation)
	{
		if (myid == 2)
		{
			/*
			进程2读取验证集并将属性信息和真实类标分别储存
			*/
			ifstream infile;
			infile.open(validation_file_name);
			string s;
			int cnt = 0;
			while (getline(infile, s))
			{
				stringstream ss(s);
				string a;
				while (getline(ss, a, ','))
				{
					if (cnt % (dim + 1) == 0) Val_label_real[cnt / (dim + 1)] = atoi(a.c_str());
					else Data_val[cnt - (cnt / (dim + 1)) - 1] = atof(a.c_str());
					cnt++;
				}
			}
			infile.close();
		}
	}

	MPI_Bcast(Data_train, N_train * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD); //进程0广播训练集属性信息
	MPI_Bcast(Train_label, N_train, MPI_INT, 0, MPI_COMM_WORLD); //进程0广播训练集类标信息
	MPI_Scatter(Data_test, batch_test * dim, MPI_DOUBLE, Data_test_buffer, batch_test * dim, MPI_DOUBLE, 1, MPI_COMM_WORLD); //进程1分发测试集属性信息
	if(Validation) MPI_Scatter(Data_val, batch_val* dim, MPI_DOUBLE, Data_val_buffer, batch_val* dim, MPI_DOUBLE, 2, MPI_COMM_WORLD); //进程2分发验证集属性信息

	if (Normalize)
	{
		/*
		归一化训练集、验证集以及测试集的属性信息
		*/
		max_value = new double[dim]; //存储每一维度属性信息的全局最大值
		min_value = new double[dim]; //存储每一维度属性信息的全局最小值

		max_value_buffer = new double[dim]; //存储每一维度属性信息在当前进程的最大值
		min_value_buffer = new double[dim]; //存储每一维度属性信息在当前进程的最小值
		for (int i = 0; i < dim; i++)
		{
			max_value_buffer[i] = -1;
			min_value_buffer[i] = 999999;
		}

		for (int i = 0; i < batch_train; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				double data = Data_train[(myid * batch_train * dim) + (i * dim) + j];
				if (data > max_value_buffer[j]) max_value_buffer[j] = data;
				if (data < min_value_buffer[j]) min_value_buffer[j] = data;
			}
		}
		for (int i = 0; i < batch_test; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				double data = Data_test_buffer[i * dim + j];
				if (data > max_value_buffer[j]) max_value_buffer[j] = data;
				if (data < min_value_buffer[j]) min_value_buffer[j] = data;
			}
		}
		if (Validation)
		{
			for (int i = 0; i < batch_val; i++)
			{
				for (int j = 0; j < dim; j++)
				{
					double data = Data_val_buffer[i * dim + j];
					if (data > max_value_buffer[j]) max_value_buffer[j] = data;
					if (data < min_value_buffer[j]) min_value_buffer[j] = data;
				}
			}
		}

		MPI_Allreduce(max_value_buffer, max_value, dim, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); //统计每一维度属性信息的全局最大值
		MPI_Allreduce(min_value_buffer, min_value, dim, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); //统计每一维度属性信息的全局最小值

		for (int i = 0; i < N_train; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				int aim = i * dim + j;
				if (max_value[j] - min_value[j] != 0) Data_train[aim] = (Data_train[aim] - min_value[j]) / (max_value[j] - min_value[j]);
			}
		}
		for (int i = 0; i < batch_test; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				int aim = i * dim + j;
				if (max_value[j] - min_value[j] != 0) Data_test_buffer[aim] = (Data_test_buffer[aim] - min_value[j]) / (max_value[j] - min_value[j]);
			}
		}
		if (Validation)
		{
			for (int i = 0; i < batch_val; i++)
			{
				for (int j = 0; j < dim; j++)
				{
					int aim = i * dim + j;
					if (max_value[j] - min_value[j] != 0) Data_val_buffer[aim] = (Data_val_buffer[aim] - min_value[j]) / (max_value[j] - min_value[j]);
				}
			}
		}
	}

	if (Validation)
	{
		/*
		对验证集的数据运行KNN
		*/
		train_data_dis* d1;
		d1 = new train_data_dis[N_train];
		for (int i = 0; i < batch_val; i++)
		{
			for (int j = 0; j < N_train; j++)
			{
				d1[j].label = Train_label[j];
				if (Euclidean_distance) d1[j].dis = Euclidean_D(i, j, dim, Data_train, Data_val_buffer);
				else d1[j].dis = Manhattan_D(i, j, dim, Data_train, Data_val_buffer);
			}
			sort(d1, d1 + N_train, Comp);
			double max_cnt = 0; int max_label = -1;
			double* label_cnt;
			label_cnt = new double[class_cnt];
			for (int j = 0; j < class_cnt; j++) label_cnt[j] = 0;
			for (int j = 0; j < K; j++)
			{
				label_cnt[d1[j].label] ++;
				if (label_cnt[d1[j].label] > max_cnt)
				{
					max_label = d1[j].label;
					max_cnt = label_cnt[d1[j].label];
				}
			}
			Val_label_buffer[i] = max_label;
		}

		MPI_Gather(Val_label_buffer, batch_val, MPI_INT, Val_label, batch_val, MPI_INT, 2, MPI_COMM_WORLD); //统计全局验证集的类标至进程2

		if (myid == 2)
		{
			/*
			在进程2中计算验证集预测准确率
			*/
			double acc = acc_calc(Val_label_real, Val_label, N_val);
			cout << "accuracy = " << acc << endl;
		}
	}

	/*
	对测试集的数据运行KNN
	*/
	train_data_dis* d2;
	d2 = new train_data_dis[N_train];

	for (int i = 0; i < batch_test; i++)
	{
		for (int j = 0; j < N_train; j++)
		{
			d2[j].label = Train_label[j];
			if (Euclidean_distance) d2[j].dis = Euclidean_D(i, j, dim, Data_train, Data_test_buffer);
			else d2[j].dis = Manhattan_D(i, j, dim, Data_train, Data_test_buffer);
		}
		sort(d2, d2 + N_train, Comp);
		double max_cnt = 0; int max_label = -1;
		double* label_cnt;
		label_cnt = new double[class_cnt];
		for (int j = 0; j < class_cnt; j++) label_cnt[j] = 0;
		for (int j = 0; j < K; j++)
		{
			label_cnt[d2[j].label] ++;
			if (label_cnt[d2[j].label] > max_cnt)
			{
				max_label = d2[j].label;
				max_cnt = label_cnt[d2[j].label];
			}
		}
		Test_label_buffer[i] = max_label;
	}

	MPI_Gather(Test_label_buffer, batch_test, MPI_INT, Test_label, batch_test, MPI_INT, 1, MPI_COMM_WORLD);//统计全局测试集的类标至进程1

	if (myid == 1)
	{
		/*
		在进程1中输出测试集预测类标
		*/
		ofstream outfile("Test_label.csv");
		for (int i = 0; i < N_test; i++) outfile << Test_label[i] << endl;
		outfile.close();
	}

	MPI_Barrier(MPI_COMM_WORLD);
	finish = MPI_Wtime();
	MPI_Finalize();
	if (!myid) cout << "Running time is " << (double)finish - start << " second" << endl;
}