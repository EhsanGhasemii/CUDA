#include "../include/mainwindow.h"
#include "ui_mainwindow.h"

// function2 for MSE calculation between an array and a cx_mat
double calc_mse (arma::cx_mat data1, arma::cx_mat data2, int row, int col_size) {
        /* We suppose that size of data1 is n * m and
         * sise of data 2 is 1 * m_prime and we just 
         * compaire first col_size of (row)th row of 
         * first data with second data. 
         */
        double mse = 0.0; 
        for (int i = 0; i < col_size; ++i) {
                mse += (data1(row, i).real() - data2(0, i).real()) * (data1(row, i).real() - data2(0, i).real());
                mse += (data1(row, i).imag() - data2(0, i).imag()) * (data1(row, i).imag() - data2(0, i).imag());

        }
        return mse; 
}


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    setWindowTitle("Server");
    // Reading input data =================================	
    // define batch size 
    batch_size = 1;			// batch size is for previous versions.
    data_num = 32; 
    element_len = 99;
    offset = 0;              	// 0 in cpp is equal to 1 in matlab 
    out_len = element_len - (13 - 1); 
    sigma = 1e-5; 

    // load the matrix from the CSV file
    std::string name; 
    std::string name_i; 
    std::string name_q; 

    name = "HighSNR_Target_Mask_20dBWeakerTarget_Speed_1mPerS"; 

    name_i = "./data_in/" + name + "_I.csv"; 
    y_noisy2_real.load(name_i, arma::csv_ascii); 

    name_q = "./data_in/" + name + "_Q.csv"; 
    y_noisy2_imag.load(name_q, arma::csv_ascii); 
    // End of Reading input data =============================





    // load output result of Matlab calculation to check our result
    arma::mat matlab_apc_real; 
    arma::mat matlab_apc_imag; 

    name_i = "./data_result/Result" + name + "_I.csv"; 
    matlab_apc_real.load(name_i, arma::csv_ascii); 

    name_q = "./data_result/Result" + name + "_Q.csv"; 
    matlab_apc_imag.load(name_q, arma::csv_ascii); 

    matlab_apc = arma::cx_mat(matlab_apc_real, matlab_apc_imag);  // gpu apc shape: data_num * out_size: (99 * 87)


    server = new QTcpServer(this);
    connect(server,&QTcpServer::newConnection,this,&MainWindow::clientAppend);
    server->listen(QHostAddress::LocalHost, 50005);

    timer = new QTimer(this);
    connect(timer,&QTimer::timeout,this,&MainWindow::timerTrigged);
    timer->start(1000);
}

MainWindow::~MainWindow()
{
    delete ui;
    while (clientList.length() > 0) {
        clientList.at(0)->close();
        clientList.at(0)->abort();
        clientList.removeFirst();
    }

    server->close();
    timer->stop();
    timer->deleteLater();
    server->deleteLater();
}

void MainWindow::clientAppend()
{
    QTcpServer *temp = (QTcpServer*)(sender());
    QTcpSocket *client = temp->nextPendingConnection();
    connect(client,&QTcpSocket::disconnected,this,&MainWindow::clientRemoved);
    client->setReadBufferSize(1e6);
    clientList.append(client);
    ui->lbl_clientCounter->setText(QString::number(clientList.length()));
    qDebug() << "Client Added";
}

void MainWindow::clientRemoved()
{
    QTcpSocket *temp = (QTcpSocket*)(sender());
    clientList.removeOne(temp);
    ui->lbl_clientCounter->setText(QString::number(clientList.length()));
    qDebug() << "Client Removed";
}

void MainWindow::timerTrigged()
{
    if(clientList.length() > 0)
    {
        static quint16 rng = 0;
        static quint16 az  = 0;
        static quint16 elv = 0;

        rng++;

        az++;
        if(az == 360)
            az = 0;

        elv++;
        if(elv == 180)
            elv = 0;

	// Copy variables from local to the server
        serverData.range        = rng;
        serverData.azimuth      = az ;
        serverData.elevation    = elv;

	serverData.file_num = 1; 
	serverData.chunk_num = rng % int(std::ceil(element_len / (data_num * 1.0))); 
	serverData.step = rng; 
	
	serverData.batch_size = batch_size; 
	serverData.data_num = std::min(data_num, element_len - serverData.chunk_num * data_num); 
	serverData.offset = offset; 
	serverData.element_len = element_len; 
	serverData.out_len = out_len; 
	serverData.sigma = sigma;

	for (int i = 0; i < serverData.data_num; ++i) { 
		for (int j = 0; j < element_len; ++j) { 
			serverData.data_I[i * element_len + j] = y_noisy2_real(i + serverData.chunk_num * data_num, j); 
			serverData.data_Q[i * element_len + j] = y_noisy2_imag(i + serverData.chunk_num * data_num, j); 
		}
	}
	// ==================================================

	// define data to store receive data from the client back
	QByteArray receiveData;

        for (int i = 0; i < clientList.length(); ++i)
        {
            if(clientList.at(i)->bytesAvailable() > 0)
            {
                quint16 rHeader = 0;

		// receive result data from the client
		receiveData.append(clientList.at(i)->readAll());
                QByteArray data2  = receiveData.mid(0,sizeof(ClientData));
                ClientData cData = *reinterpret_cast<ClientData*>(data2.data());
                receiveData.remove(0,sizeof(ClientData));

		if (messageTimes.contains(cData.step)) { 
			int timeDiff = messageTimes.value(cData.step).elapsed();
			std::cout << "(Id, time): " << cData.step << " " << timeDiff << std::endl;

			arma::mat real_part(cData.data_I, cData.out_len, cData.data_num);
			arma::mat imag_part(cData.data_Q, cData.out_len, cData.data_num); 

			gpu_apc = arma::cx_mat(real_part, imag_part); 
			gpu_apc = gpu_apc.st(); 

			// check the GPU results that received from the client with matlab results
			my_indx = 0;
			while (my_indx < gpu_apc.n_rows) { 
				double mse = calc_mse(matlab_apc, gpu_apc.row(my_indx), my_indx + cData.chunk_num * data_num, out_len); 
				
				std::cout << "my_indx: " << my_indx << std::endl; 
				std::cout << "MSE: " << mse << std::endl; 
				std::cout << "----------" << std::endl;

				my_indx ++; // go to the next data row
			}



		}

                rHeader = cData.header;
                if(rHeader == 0xFA15)
                {
                    clientList.at(i)->write((char*)&serverData,sizeof (serverData));
		    QTime time; 
		    time.start(); 
		    messageTimes.insert(serverData.step, time); 
                }
            }
        }

        ui->lbl_rng->setText(QString::number(rng));
        ui->lbl_az->setText(QString::number(az))  ;
        ui->lbl_elv->setText(QString::number(elv));
	
	// Main body of the Algorithm =======
	std::cout << "file_name: " << serverData.file_num << std::endl; 
	std::cout << "chunk_num: " << serverData.chunk_num << std::endl; 
	std::cout << "step: " << serverData.step << std::endl;

	std::cout << "batch_size: " << serverData.batch_size << std::endl; 
	std::cout << "data_num: " << serverData.data_num << std::endl; 
	std::cout << "offset: " << serverData.offset << std::endl; 
	std::cout << "element_len: " << serverData.element_len << std::endl; 
	std::cout << "out_len: " << serverData.out_len << std::endl; 
	std::cout << "simga: " << serverData.sigma << std::endl;
	std::cout << "test: " << std::ceil(99 / (32 * 1.0)) << std::endl; 
	std::cout << "===========================" << std::endl; 

	// End of the algorithm body ========


    }
}

