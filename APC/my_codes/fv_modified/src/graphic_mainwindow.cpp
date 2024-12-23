#include "../include/graphic_mainwindow.h"
#include "ui_graphic_mainwindow.h"

#include <QDebug>
#include <QApplication>
#include <armadillo>
#include <vector>

using namespace arma; 

double calc_mse (cx_mat data1, cx_mat data2, int row, int col_size);

cx_mat s;
cx_mat y_noisy; 
// modifying ======================
cx_mat y_noisy2;
cx_mat gpu_apc;
cx_mat gpu_test_apc; 
int my_indx;
int data_num; 
int batch_size; 
mat alpha; 
// ================================


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Populate the QComboBox with alpha values
    ui->comboBox->addItem("1");
    ui->comboBox->addItem("2");
    ui->comboBox->addItem("3");

    // Ensure single connection
    disconnect(ui->submitButton, &QPushButton::clicked, this, &MainWindow::on_submitButton_clicked);
    connect(ui->submitButton, &QPushButton::clicked, this, &MainWindow::on_submitButton_clicked);

    // Debug statement to ensure connection is established only once
    qDebug() << "Button signal connected.";
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_submitButton_clicked()
{
    // Get the selected alpha value
    int a_size = ui->comboBox->currentText().toInt();

    // Process the alpha value
    processAlpha(a_size);

    // Print the alpha value to the terminal once
    qDebug() << "Selected alpha value:" << a_size;

    // Terminate the application
    QApplication::quit();
}

// Define the function to process the alpha value
void MainWindow::processAlpha(int a_size)
{

	// define batch size 
	batch_size = 1;				 // batch size is for previous versions.
	int offset = 0;              // 0 in cpp is equal to 1 in matlab 
	int element_len = 98;
	int out_len = element_len - (13 - 1); 
	double sigma = 1e-5;

	// load the matrix from the CSV files
	std::vector<arma::cx_mat> my_vec;
	//arma::cx_mat y_noisy2;
	arma::mat y_noisy2_real;
	arma::mat y_noisy2_imag; 

	std::string name;
	std::string name_i; 
	std::string name_q; 

	name = "HighSNR_Target_Mask_20dBWeakerTarget_Speed_1mPerS"; 

	name_i = "./data_in/" + name + "_I.csv"; 
	y_noisy2_real.load(name_i, arma::csv_ascii); 

	name_q = "./data_in/" + name + "_Q.csv"; 
	y_noisy2_imag.load(name_q, arma::csv_ascii); 

	y_noisy2 = arma::cx_mat(y_noisy2_real, y_noisy2_imag) / std::pow(2.0, 10);
	y_noisy2 = y_noisy2.st();

	y_noisy2 = y_noisy2.submat(offset, 0, offset + element_len, y_noisy2.n_cols - 1); 
	//y_noisy2 = y_noisy2.submat(0, 0, y_noisy2.n_rows-1, 0); 

	my_vec.push_back(y_noisy2); 

	// start coding CUDA =============================
    int temp=0;
    s.resize(13,1);
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=-1;
    s(temp++,0)=-1;
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=-1;
    s(temp++,0)=1;
    s(temp++,0)=-1;
    s(temp++,0)=1;


	////
	alpha = mat(a_size, 1); 
// 	alpha(0,0) = 1.9;
// 	alpha(1,0) = 1.8;

  for (uint32_t i = 0; i < a_size; ++i) { 
    alpha(i, 0) = 1.9 - 0.1 * i; 
  }


	// allocate memory in CPU for calculation
	float* y_n_real;
	float* y_n_imag; 
	double* X_real; 
	double* X_imag;
	//double* rho_real; 
	//double* rho_imag; 
	float* R_real; 
	float* R_imag;
	float* Ss_real; 
	float* Ss_imag; 
	float* s_real; 
	float* s_imag;
	float* alpha_real; 

	double* output_real;
	double* output_imag;

//	double* test_real;
//	double* test_imag; 

	// define our variables
	data_num = y_noisy2.n_cols; 
	int y_n_size; 
	int X_size; 
	int R_row; 
	int Ss_size;
	int s_size;
	int alpha_size = a_size; 
	int print_flag = 1;						// print report from GPU function
	int num_iter = 1000; 

	fun1(s, 
		 my_vec, 
		 13, 
		 alpha, 
		 1e-5, 
		 &y_n_real, 
		 &y_n_imag, 
		 &X_real, 
		 &X_imag, 
		 &R_real, 
		 &R_imag,
		 &Ss_real,
		 &Ss_imag,
		 &s_real,
		 &s_imag,
		 &alpha_real,
		 &output_real,
		 &output_imag, 

		 batch_size, 
		 data_num + 1,
		 y_n_size,
		 X_size, 
		 R_row, 
		 Ss_size, 
		 s_size, 
		 alpha_size, 
		 num_iter
		 );

}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}


// function2 for MSE calculation between an array and a cx_mat
double calc_mse (cx_mat data1, cx_mat data2, int row, int col_size) {
	/* We suppose that size of data1 is n * m and
	 * sise of data 2 is 1 * m_prime and we just 
	 * compaire first col_size of (row)th row of 
	 * first data with second data. 
	 */
	double mse = 0.0; 
	for (int i = 0; i < col_size; ++i) {
		mse += (data1(row, i).real() - data2(0, i).real()) * (data1(row, i).real() - data2(0, i).real());
		mse += (data1(row, i).imag() - data2(0, i).imag()) * (data1(row, i).imag() - data2(0, i).imag());

//		std::cout << data1(row, i).real() << " " << data1(row, i).imag() << std::endl; 
//		std::cout << data2(0, i).real() << " " << data2(0, i).imag() << std::endl; 
//		std::cout << "mse: " << mse << std::endl; 
//		std::cout << "------------------------" << std::endl; 

	}
	return mse; 
}

