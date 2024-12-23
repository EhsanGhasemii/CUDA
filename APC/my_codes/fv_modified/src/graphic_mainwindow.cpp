#include "../include/graphic_mainwindow.h"
#include "ui_graphic_mainwindow.h"
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Populate the QComboBox with alpha values
    ui->comboBox->addItem("1");
    ui->comboBox->addItem("2");
    ui->comboBox->addItem("3");

    // Connect the QPushButton clicked signal to the slot
    connect(ui->submitButton, &QPushButton::clicked, this, &MainWindow::on_submitButton_clicked);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_submitButton_clicked()
{
    // Get the selected alpha value
    QString alpha = ui->comboBox->currentText();

    // Print the alpha value to the terminal
    qDebug() << "Selected alpha value:" << alpha;
}


