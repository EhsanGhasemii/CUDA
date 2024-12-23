#include "../include/graphic_mainwindow.h"
#include "ui_graphic_mainwindow.h"

#include <QDebug>
#include <QApplication>

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
    int alpha = ui->comboBox->currentText().toInt();

    // Process the alpha value
    processAlpha(alpha);

    // Print the alpha value to the terminal once
    qDebug() << "Selected alpha value:" << alpha;

    // Terminate the application
    QApplication::quit();
}

// Define the function to process the alpha value
void MainWindow::processAlpha(int alpha)
{
    switch(alpha)
    {
        case 1:
            qDebug() << "Processing alpha = 1: Performing action for alpha 1.";
            break;
        case 2:
            qDebug() << "Processing alpha = 2: Performing action for alpha 2.";
            break;
        case 3:
            qDebug() << "Processing alpha = 3: Performing action for alpha 3.";
            break;
        default:
            qDebug() << "Unknown alpha value.";
            break;
    }
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}

