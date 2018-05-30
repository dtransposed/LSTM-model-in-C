#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
struct LSTM_parameters{

	    double **LSTM1_c_U;
		double **LSTM1_c_W;
		double **LSTM1_c_b;

		double **LSTM1_f_U;
		double **LSTM1_f_W;
		double **LSTM1_f_b;

		double **LSTM1_i_U;
		double **LSTM1_i_W;
		double **LSTM1_i_b;

		double **LSTM1_o_U;
		double **LSTM1_o_W;
		double **LSTM1_o_b;

		double **LSTM2_c_U;
		double **LSTM2_c_W;
		double **LSTM2_c_b;

		double **LSTM2_f_U;
		double **LSTM2_f_W;
		double **LSTM2_f_b;

		double **LSTM2_i_U;
		double **LSTM2_i_W;
		double **LSTM2_i_b;

		double **LSTM2_o_U;
		double **LSTM2_o_W;
		double **LSTM2_o_b;

		double **LSTM3_c_U;
		double **LSTM3_c_W;
		double **LSTM3_c_b;

		double **LSTM3_f_U;
		double **LSTM3_f_W;
		double **LSTM3_f_b;

		double **LSTM3_i_U;
		double **LSTM3_i_W;
		double **LSTM3_i_b;

		double **LSTM3_o_U;
		double **LSTM3_o_W;
		double **LSTM3_o_b;

		double **W_dense;
		double **b_dense;

	};
struct LSTM_layer{

	double **state_of_the_layer;	// Placeholder for a cell state of a layer.
	double **output_from_the_layer;	//Placeholder for an output of a layer.

};
void print_list (int seq_len, double *matrix){ // Print a matrix (used for visualization)

	for (int i = 0; i < seq_len; i++){

        printf("%f ", matrix[i]);

		}

	printf("\n\n");

}
void shift_sequence (double new_value, double seq_len, double *output, double *input){

    output[0] = new_value;

    for (int i = 1; i < seq_len; i++){

        output[i] = input [i-1];

    }
}
void copy_list (double seq_len, double * output, double *input){

        for (int i = 0; i < seq_len; i++){

            output[i] = input[i];

        }
}
void input_value_to_vector (double new_value, int timesteps, double *result, double *sequence){

    shift_sequence(new_value, timesteps, result, sequence);
    copy_list(timesteps, sequence, result);

}
int read_matrices (char txt_file[], struct LSTM_parameters *NN_model, int no_of_units ){ // Write numerical data from .txt file into a pre-defined matrix.

	FILE * file = fopen(txt_file, "r");

	double params[4]; int no_parameters = 4;

	for (int i = 0 ;i < no_parameters; i++){

		fscanf(file, "%lf", &params[i]);
	}

	// 4 by 4 matrices
	NN_model->LSTM1_c_U = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM1_f_U = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM1_i_U = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM1_o_U = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM2_c_U = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM2_f_U = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM2_i_U = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM2_o_U = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM3_c_U = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM3_f_U = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM3_i_U = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM3_o_U = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM2_c_W = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM2_f_W = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM2_i_W = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM2_o_W = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM3_c_W = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM3_f_W = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM3_i_W = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM3_o_W = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	for (int i = 0; i < no_of_units; i++){

		NN_model->LSTM1_c_U[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM1_f_U[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM1_i_U[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM1_o_U[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM2_c_U[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM2_f_U[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM2_i_U[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM2_o_U[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM3_c_U[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM3_f_U[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM3_i_U[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM3_o_U[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
	    NN_model->LSTM2_c_W[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM2_f_W[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM2_i_W[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM2_o_W[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM3_c_W[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM3_f_W[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM3_i_W[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM3_o_W[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
	}


	NN_model->LSTM1_f_W = (double **)calloc(1,1*sizeof(double));
	NN_model->LSTM1_i_W = (double **)calloc(1,1*sizeof(double));
	NN_model->LSTM1_o_W = (double **)calloc(1,1*sizeof(double));
	NN_model->LSTM1_c_W = (double **)calloc(1,1*sizeof(double));
	for (int i = 0; i < 1; i++){

		NN_model->LSTM1_f_W[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM1_i_W[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM1_o_W[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		NN_model->LSTM1_c_W[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));

	}


	NN_model->b_dense = (double **)calloc(1,1*sizeof(double));
	for (int i = 0; i < 1; i++){

		NN_model->b_dense[i] = (double *)calloc(1,1*sizeof(double));

	}


	NN_model->LSTM1_f_b = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM1_i_b = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM1_o_b = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM1_c_b = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM2_f_b = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM2_i_b = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM2_o_b = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM2_c_b = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM3_f_b = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM3_i_b = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM3_o_b = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->LSTM3_c_b = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	NN_model->W_dense = (double **)calloc(no_of_units,no_of_units*sizeof(double));
	for (int i = 0; i < no_of_units; i++){

		NN_model->LSTM1_f_b[i] = (double *)calloc(1,1*sizeof(double));
		NN_model->LSTM1_i_b[i] = (double *)calloc(1,1*sizeof(double));
		NN_model->LSTM1_o_b[i] = (double *)calloc(1,1*sizeof(double));
		NN_model->LSTM1_c_b[i] = (double *)calloc(1,1*sizeof(double));
		NN_model->LSTM2_f_b[i] = (double *)calloc(1,1*sizeof(double));
		NN_model->LSTM2_i_b[i] = (double *)calloc(1,1*sizeof(double));
		NN_model->LSTM2_o_b[i] = (double *)calloc(1,1*sizeof(double));
		NN_model->LSTM2_c_b[i] = (double *)calloc(1,1*sizeof(double));
		NN_model->LSTM3_f_b[i] = (double *)calloc(1,1*sizeof(double));
		NN_model->LSTM3_i_b[i] = (double *)calloc(1,1*sizeof(double));
		NN_model->LSTM3_o_b[i] = (double *)calloc(1,1*sizeof(double));
		NN_model->LSTM3_c_b[i] = (double *)calloc(1,1*sizeof(double));
		NN_model->W_dense[i] = (double *)calloc(1,1*sizeof(double));

	}

	for (int i = 0 ;i < 1; i++){
		for (int j = 0 ;j < 4; j++){
			fscanf(file, "%lf", &NN_model->LSTM1_i_W[i][j]);
		}
	}



	for (int i = 0 ;i < 1; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM1_f_W[i][j]);
		}
	}


	for (int i = 0 ;i < 1; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM1_c_W[i][j]);
		}
	}


	for (int i = 0 ;i < 1; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM1_o_W[i][j]);
		}
	}

	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM1_i_U[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM1_f_U[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM1_c_U[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM1_o_U[i][j]);
		}
	}

	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->LSTM1_i_b[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->LSTM1_f_b[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->LSTM1_c_b[i][j]);
		}
	}

	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->LSTM1_o_b[i][j]);
		}
	}

	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM2_i_W[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM2_f_W[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM2_c_W[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM2_o_W[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM2_i_U[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM2_f_U[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM2_c_U[i][j]);
		}
	}

	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM2_o_U[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->LSTM2_i_b[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->LSTM2_f_b[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->LSTM2_c_b[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->LSTM2_o_b[i][j]);
		}
	}

		for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM3_i_W[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM3_f_W[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM3_c_W[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM3_o_W[i][j]);
		}
	}

	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM3_i_U[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM3_f_U[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM3_c_U[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < no_of_units; j++){
			fscanf(file, "%lf", &NN_model->LSTM3_o_U[i][j]);
		}
	}

	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->LSTM3_i_b[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->LSTM3_f_b[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->LSTM3_c_b[i][j]);
		}
	}


	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->LSTM3_o_b[i][j]);
		}
	}

	for (int i = 0 ;i < no_of_units; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->W_dense[i][j]);
		}
	}

	for (int i = 0 ;i < 1; i++){
		for (int j = 0 ;j < 1; j++){
			fscanf(file, "%lf", &NN_model->b_dense[i][j]);
		}
	}

	fclose(file);
	return 0;

}
void read_parameters (char txt_file[], double parameters[4] ){ // Write numerical data from .txt file into a pre-defined matrix.

	FILE * file = fopen(txt_file, "r");

	for (int i = 0 ;i < 4; i++){

		fscanf(file, "%lf", &parameters[i]);

	}

	fclose(file);
}
void reset_variables (int size_rows, int size_columns, struct LSTM_layer *layer){ //Sets LSTM state and hidden output to zero.

	for (int i = 0; i < size_rows; i++){

		for (int j = 0; j < size_columns; j++){

			layer->state_of_the_layer[i][j] = 0;
			layer->output_from_the_layer[i][j] = 0;

		}
	}
}
void hard_sigmoid (int size_rows, int size_columns, double **matrix){ // Calculate hard_sigmoid of an input.

	for (int i = 0; i < size_rows; i++){

		for (int j = 0; j < size_columns; j++){

			if (matrix[i][j]*0.2+0.5 > 1){

				matrix[i][j] = 1;
			}

			else{

				matrix[i][j] = matrix[i][j] * 0.2 + 0.5;

			}
		}
	}

	for (int i = 0; i < size_rows; i++){

		for (int j = 0; j < size_columns; j++){

			if (matrix[i][j] < 0){

				matrix[i][j] = 0;

			}


		}
	}
}
void tanh_funct (int size_rows, int size_columns, double **matrix){ // Calculate tanh of an input.

	for (int i = 0; i < size_rows; i++){

		for (int j = 0; j < size_columns; j++){

			matrix[i][j] = tanh(matrix[i][j]);

		}
	}
}
void matrix_mult (int rowFirst, int columnFirst, int columnSecond, double **product, double **firstMatrix, double **secondMatrix){ // Multiplication of two matrices.

		for (int i = 0; i < rowFirst; i++){

			for (int j = 0; j < columnSecond; j++){

				product[i][j] = 0;
			}
		}

		for (int i = 0; i < rowFirst; i++){

			for (int j = 0; j < columnSecond; j++){

				for (int k = 0; k < columnFirst; k++){

					product[i][j] += firstMatrix[i][k]*secondMatrix[k][j];
				}
			}
		}
}
void print_matrix (int size_rows, int size_columns, double **matrix){ // Print a matrix (used for visualization)

	for (int i = 0; i < size_rows; i++){

		for (int j = 0; j < size_columns; j++){

			printf("%f ",matrix[i][j]);

		}

	printf("\n");

	}
}
void matrix_hadamard(int size_rows,int size_columns, double **result, double **matrix1, double **matrix2){ // Entrywise product (Hadamard or Schur product) of two matrices.

	for (int i = 0; i < size_rows; i++){

		for (int j = 0; j < size_columns; j++){

			result[i][j] = matrix1[i][j]*matrix2[i][j];
		}
	}

}
void broadcast_array(int size_rows, int size_columns, int size_rows_input, int size_columns_input, double **result, double **input_matrix){ // Broadcast shapes of an array across the larger array so that they have compatible shapes.

	for (int i = 0; i < size_rows; i++){
		for (int j = 0; j < size_columns; j++){
			result[i][j] = input_matrix[0][j];
		}
	}

}
void broadcast_array_bias(int size_rows, int size_columns, int size_rows_input, int size_columns_input, double **result, double **input_matrix){ // Broadcast shapes of an array across the larger array so that they have compatible shapes.

	for (int i = 0; i < size_columns; i++){

		for (int j = 0; j < size_rows; j++){

			result[i][j] = input_matrix[i][0];
		}
	}

}
void matrix_sum (int size_rows,int size_columns, double **result, double **firstMatrix, double **secondMatrix){ // Matrix addition.

	for (int i = 0; i < size_rows; i++){
		for (int j = 0; j < size_columns; j++){

			result[i][j] = firstMatrix[i][j]+secondMatrix[i][j];
		}
	}
}
void LSTM_first_cell(int input_rows, int input_columns,int no_of_units, int timesteps, struct LSTM_layer *layer, double **x, struct LSTM_parameters NN_model){


	double **placeholder0= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder1= (double **)calloc(input_rows,input_rows*sizeof(double));
	double **placeholder2= (double **)calloc(input_rows,input_rows*sizeof(double));
	double **placeholder3= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder4= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder5= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **forget_gate= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **input_gate= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **output_gate= (double **)calloc(timesteps,timesteps*sizeof(double));

	for (int i = 0; i < input_rows; i++){
		placeholder1[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder2[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
	}

	for (int i = 0; i < timesteps; i++){
		placeholder0[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder3[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder4[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder5[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		forget_gate[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		input_gate[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		output_gate[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
	}


	matrix_mult ( input_rows, input_columns, no_of_units, placeholder1, x, NN_model.LSTM1_f_W);
	matrix_mult ( timesteps, no_of_units, no_of_units, forget_gate, layer->output_from_the_layer, NN_model.LSTM1_f_U);
	broadcast_array ( timesteps, no_of_units, input_rows, no_of_units, placeholder0, placeholder1);
	matrix_sum ( timesteps, no_of_units, forget_gate, forget_gate, placeholder0);
	broadcast_array_bias ( timesteps, no_of_units, input_rows, no_of_units, placeholder0, NN_model.LSTM1_f_b);
	matrix_sum ( timesteps, no_of_units, forget_gate, forget_gate, placeholder0);
	hard_sigmoid ( timesteps, no_of_units, forget_gate);



	matrix_mult ( input_rows, input_columns, no_of_units, placeholder1, x, NN_model.LSTM1_i_W);
	matrix_mult ( timesteps, no_of_units, no_of_units, input_gate, layer->output_from_the_layer, NN_model.LSTM1_i_U);
	broadcast_array ( timesteps, no_of_units, input_rows, no_of_units, placeholder0, placeholder1);
	matrix_sum ( timesteps, no_of_units, input_gate, input_gate, placeholder0);
	broadcast_array_bias ( timesteps, no_of_units, input_rows, no_of_units, placeholder0, NN_model.LSTM1_i_b);
	matrix_sum ( timesteps, no_of_units, input_gate, input_gate, placeholder0);
	hard_sigmoid ( timesteps, no_of_units, input_gate);


	matrix_mult ( input_rows, input_columns, no_of_units, placeholder1, x, NN_model.LSTM1_o_W);
	matrix_mult ( timesteps, no_of_units, no_of_units, output_gate, layer->output_from_the_layer, NN_model.LSTM1_o_U);
	broadcast_array ( timesteps, no_of_units, input_rows, no_of_units, placeholder0, placeholder1);
	matrix_sum ( timesteps, no_of_units, output_gate, output_gate , placeholder0);
	broadcast_array_bias (timesteps, no_of_units, input_rows, no_of_units, placeholder0, NN_model.LSTM1_o_b);
	matrix_sum ( timesteps, no_of_units, output_gate, output_gate, placeholder0);
	hard_sigmoid ( timesteps, no_of_units, output_gate);


	matrix_mult ( timesteps, no_of_units, no_of_units, placeholder4, layer->output_from_the_layer, NN_model.LSTM1_c_U);
	matrix_mult (input_rows, input_columns, no_of_units, placeholder1, x, NN_model.LSTM1_c_W);
	broadcast_array ( timesteps, no_of_units, input_rows, no_of_units, placeholder0, placeholder1);
	matrix_sum ( timesteps, no_of_units, placeholder4, placeholder4, placeholder0);
	broadcast_array_bias (timesteps, no_of_units, input_rows, no_of_units, placeholder3, NN_model.LSTM1_c_b);
	matrix_sum ( timesteps, no_of_units, placeholder4, placeholder4, placeholder3);
	tanh_funct ( timesteps, no_of_units, placeholder4);
	matrix_hadamard ( timesteps, no_of_units, placeholder4, input_gate, placeholder4);
	matrix_hadamard ( timesteps, no_of_units, placeholder0, forget_gate, layer->state_of_the_layer);
	matrix_sum (timesteps, no_of_units, layer->state_of_the_layer, placeholder4, placeholder0);

	for (int i = 0; i < timesteps; i++){

		for (int j = 0;j < no_of_units; j++){

			placeholder5[i][j]=layer->state_of_the_layer[i][j];

		}
	}

	tanh_funct ( timesteps, no_of_units, placeholder5);
	matrix_hadamard ( timesteps, no_of_units, layer->output_from_the_layer, output_gate, placeholder5);
}
void LSTM_second_cell(int no_of_units, int timesteps, struct LSTM_layer *previous_layer, struct LSTM_layer *layer, struct LSTM_parameters NN_model) {


	/*Calculation of the forget gate */

	double **placeholder5= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder6= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder7= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder8= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder9= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder10= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **input_gate= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **output_gate= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **forget_gate= (double **)calloc(timesteps,timesteps*sizeof(double));

	for (int i = 0; i < timesteps; i++){
		placeholder5[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder6[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder7[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder8[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder9[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder10[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		input_gate[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		output_gate[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		forget_gate[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));

	}


	matrix_mult ( timesteps, no_of_units, no_of_units, placeholder5, previous_layer->output_from_the_layer, NN_model.LSTM2_f_W);
	matrix_mult ( timesteps, no_of_units, no_of_units, forget_gate, layer->output_from_the_layer ,NN_model.LSTM2_f_U);
	matrix_sum ( timesteps, no_of_units, forget_gate, placeholder5, forget_gate);
	broadcast_array_bias ( timesteps, no_of_units, 1, no_of_units, placeholder6, NN_model.LSTM2_f_b);
	matrix_sum ( timesteps, no_of_units, forget_gate, forget_gate, placeholder6);
	hard_sigmoid (timesteps, no_of_units, forget_gate);

	//print_matrix(timesteps,no_of_units,forget_gate);

	matrix_mult ( timesteps, no_of_units, no_of_units, placeholder5, previous_layer->output_from_the_layer, NN_model.LSTM2_i_W);
	matrix_mult ( timesteps, no_of_units, no_of_units, input_gate, layer->output_from_the_layer, NN_model.LSTM2_i_U);
	matrix_sum ( timesteps, no_of_units, input_gate, placeholder5, input_gate);
	broadcast_array_bias ( timesteps, no_of_units, 1, no_of_units, placeholder6, NN_model.LSTM2_i_b);
	matrix_sum ( timesteps, no_of_units, input_gate, input_gate, placeholder6);
	hard_sigmoid ( timesteps, no_of_units, input_gate);

	//print_matrix(timesteps,no_of_units,input_gate);

	matrix_mult ( timesteps, no_of_units, no_of_units, placeholder5, previous_layer->output_from_the_layer, NN_model.LSTM2_o_W);
	matrix_mult ( timesteps, no_of_units, no_of_units, output_gate, layer->output_from_the_layer, NN_model.LSTM2_o_U);
	matrix_sum ( timesteps, no_of_units, output_gate, placeholder5, output_gate);
	broadcast_array_bias ( timesteps, no_of_units, 1, no_of_units, placeholder6, NN_model.LSTM2_o_b);
	matrix_sum ( timesteps, no_of_units, output_gate, output_gate, placeholder6);
	hard_sigmoid ( timesteps, no_of_units, output_gate);

	//print_matrix(timesteps,no_of_units,output_gate);

	matrix_mult ( timesteps, no_of_units, no_of_units, placeholder9, layer->output_from_the_layer, NN_model.LSTM2_c_U);
	matrix_mult ( timesteps, no_of_units, no_of_units, placeholder7, previous_layer->output_from_the_layer, NN_model.LSTM2_c_W);
	broadcast_array_bias ( timesteps, no_of_units, no_of_units, no_of_units, placeholder8, NN_model.LSTM2_c_b);
	matrix_sum ( timesteps, no_of_units, placeholder9, placeholder9, placeholder7);
	matrix_sum ( timesteps, no_of_units, placeholder9, placeholder9, placeholder8);
	tanh_funct ( timesteps, no_of_units, placeholder9);
	matrix_hadamard ( timesteps, no_of_units, placeholder9, input_gate, placeholder9);
	matrix_hadamard ( timesteps, no_of_units, placeholder8, forget_gate, layer->state_of_the_layer);
	matrix_sum ( timesteps, no_of_units, layer->state_of_the_layer, placeholder9, placeholder8);

	//print_matrix(timesteps,no_of_units,layer->state_of_the_layer);

	for (int i = 0; i < timesteps; i++){

		for (int j = 0; j < no_of_units; j++){

			placeholder10[i][j] = layer->state_of_the_layer[i][j];
		}
	}

	tanh_funct ( timesteps, no_of_units, placeholder10);
	matrix_hadamard ( timesteps, no_of_units, layer->output_from_the_layer, output_gate, placeholder10);

	}
void LSTM_third_cell(int no_of_units, int timesteps, struct LSTM_layer *previous_layer, struct LSTM_layer *layer, struct LSTM_parameters NN_model) {

	/*Calculation of the forget gate */

	double **placeholder5= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder6= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder7= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder8= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder9= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder10= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **input_gate= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **output_gate= (double **)calloc(timesteps,timesteps*sizeof(double));
	double **forget_gate= (double **)calloc(timesteps,timesteps*sizeof(double));

	for (int i = 0; i < timesteps; i++){
		placeholder5[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder6[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder7[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder8[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder9[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		placeholder10[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		input_gate[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		output_gate[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		forget_gate[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));

	}


	matrix_mult ( timesteps, no_of_units, no_of_units, placeholder5, previous_layer->output_from_the_layer, NN_model.LSTM3_f_W);
	matrix_mult ( timesteps, no_of_units, no_of_units, forget_gate, layer->output_from_the_layer ,NN_model.LSTM3_f_U);
	matrix_sum ( timesteps, no_of_units, forget_gate, placeholder5, forget_gate);
	broadcast_array_bias ( timesteps, no_of_units, 1, no_of_units, placeholder6, NN_model.LSTM3_f_b);
	matrix_sum ( timesteps, no_of_units, forget_gate, forget_gate, placeholder6);
	hard_sigmoid (timesteps, no_of_units, forget_gate);

	//print_matrix(timesteps,no_of_units,forget_gate);

	matrix_mult ( timesteps, no_of_units, no_of_units, placeholder5, previous_layer->output_from_the_layer, NN_model.LSTM3_i_W);
	matrix_mult ( timesteps, no_of_units, no_of_units, input_gate, layer->output_from_the_layer, NN_model.LSTM3_i_U);
	matrix_sum ( timesteps, no_of_units, input_gate, placeholder5, input_gate);
	broadcast_array_bias ( timesteps, no_of_units, 1, no_of_units, placeholder6, NN_model.LSTM3_i_b);
	matrix_sum ( timesteps, no_of_units, input_gate, input_gate, placeholder6);
	hard_sigmoid ( timesteps, no_of_units, input_gate);

	//print_matrix(timesteps,no_of_units,input_gate);

	matrix_mult ( timesteps, no_of_units, no_of_units, placeholder5, previous_layer->output_from_the_layer, NN_model.LSTM3_o_W);
	matrix_mult ( timesteps, no_of_units, no_of_units, output_gate, layer->output_from_the_layer, NN_model.LSTM3_o_U);
	matrix_sum ( timesteps, no_of_units, output_gate, placeholder5, output_gate);
	broadcast_array_bias ( timesteps, no_of_units, 1, no_of_units, placeholder6, NN_model.LSTM3_o_b);
	matrix_sum ( timesteps, no_of_units, output_gate, output_gate, placeholder6);
	hard_sigmoid ( timesteps, no_of_units, output_gate);

	//print_matrix(timesteps,no_of_units,output_gate);

	matrix_mult ( timesteps, no_of_units, no_of_units, placeholder9, layer->output_from_the_layer, NN_model.LSTM3_c_U);
	matrix_mult ( timesteps, no_of_units, no_of_units, placeholder7, previous_layer->output_from_the_layer, NN_model.LSTM3_c_W);
	broadcast_array_bias ( timesteps, no_of_units, no_of_units, no_of_units, placeholder8, NN_model.LSTM3_c_b);
	matrix_sum ( timesteps, no_of_units, placeholder9, placeholder9, placeholder7);
	matrix_sum ( timesteps, no_of_units, placeholder9, placeholder9, placeholder8);
	tanh_funct ( timesteps, no_of_units, placeholder9);
	matrix_hadamard ( timesteps, no_of_units, placeholder9, input_gate, placeholder9);
	matrix_hadamard ( timesteps, no_of_units, placeholder8, forget_gate, layer->state_of_the_layer);
	matrix_sum ( timesteps, no_of_units, layer->state_of_the_layer, placeholder9, placeholder8);

	//print_matrix(timesteps,no_of_units,layer->state_of_the_layer);

	for (int i = 0; i < timesteps; i++){

		for (int j = 0; j < no_of_units; j++){

			placeholder10[i][j] = layer->state_of_the_layer[i][j];
		}
	}

	tanh_funct ( timesteps, no_of_units, placeholder10);
	matrix_hadamard ( timesteps, no_of_units, layer->output_from_the_layer, output_gate, placeholder10);
	}
double dense_layer ( int timesteps, int no_of_units, struct LSTM_layer *previous_layer, struct LSTM_parameters NN_model){ // Dense layer.

	double **placeholder11 = (double **)calloc(timesteps,timesteps*sizeof(double));
	double **placeholder12 = (double **)calloc(timesteps,timesteps*sizeof(double));
	for (int i = 0; i < timesteps; i++){

		placeholder11[i] = (double *)calloc(1,1*sizeof(double));
		placeholder12[i] = (double *)calloc(1,1*sizeof(double));
	}
	double result;

	matrix_mult (timesteps, no_of_units, 1, placeholder11, previous_layer->output_from_the_layer, NN_model.W_dense);
	broadcast_array (timesteps, 1, 1, 1, placeholder12, NN_model.b_dense);
	matrix_sum (timesteps, 1, placeholder12, placeholder12, placeholder11);
	result = placeholder12[0][0];
	return result;
}
double LSTM_algorithm_three_LSTM_layers ( double *sequence, double parameters[4], struct LSTM_parameters NN_model){

	struct LSTM_layer first_layer;
	struct LSTM_layer second_layer;
	struct LSTM_layer third_layer;

    int timesteps = parameters[0];
	int no_of_batches = parameters[1];
	int no_of_layers = parameters[2];
	int no_of_units = parameters[3];

	first_layer.state_of_the_layer = (double **)calloc(timesteps,timesteps*sizeof(double));
	second_layer.state_of_the_layer = (double **)calloc(timesteps,timesteps*sizeof(double));
	third_layer.state_of_the_layer = (double **)calloc(timesteps,timesteps*sizeof(double));
	first_layer.output_from_the_layer = (double **)calloc(timesteps,timesteps*sizeof(double));
	second_layer.output_from_the_layer = (double **)calloc(timesteps,timesteps*sizeof(double));
	third_layer.output_from_the_layer = (double **)calloc(timesteps,timesteps*sizeof(double));
	for (int i = 0; i < timesteps; i++){

		first_layer.state_of_the_layer[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		second_layer.state_of_the_layer[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		third_layer.state_of_the_layer[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		first_layer.output_from_the_layer[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		second_layer.output_from_the_layer[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
		third_layer.output_from_the_layer[i] = (double *)calloc(no_of_units,no_of_units*sizeof(double));
	}

    double output;
	double **x = (double **)calloc(1,1*sizeof(double));
    for (int i = 0; i<1;i++){

        x[i] = (double **)calloc(1,1*sizeof(double));

	}

	for (int j = 0;j < timesteps; j++){
		x[0][0]=sequence[j];
		LSTM_first_cell( 1, 1, no_of_units, timesteps, &first_layer, x, NN_model );
		LSTM_second_cell(no_of_units, timesteps, &first_layer, &second_layer, NN_model);
		LSTM_third_cell(no_of_units, timesteps, &second_layer, &third_layer, NN_model);
	}

	output = dense_layer ( timesteps, no_of_units, &third_layer, NN_model);

	reset_variables(timesteps,no_of_units,&first_layer);
	reset_variables(timesteps,no_of_units,&second_layer);
	reset_variables(timesteps,no_of_units,&third_layer);

	return output;

}
int count_input(char txt_file[]){

    FILE * file2;
    int count = 1;
    char c;
    file2 = fopen(txt_file, "r");
    if(file2 == NULL)
        printf("file not found\n");
    while((c = fgetc(file2)) != EOF) {
        if(c == '\t')
            count++;
    }

	fclose(file2);
	return count;
}
void input_to_array(char txt_file[], int input_length, double *input_sequence ){ // Write numerical data from .txt file into a pre-defined matrix.

	FILE * file = fopen(txt_file, "r");

	for (int i = 0 ;i < input_length; i++){
		fscanf(file, "%lf", &input_sequence[i]);
	}
	fclose(file);
}
void append_to_file (int input_length, double *input_values, double *output_values){ // Append results to "result.txt" (semicolon delimited) file.

	FILE *pFile;

	pFile = fopen("result.txt","a");

	if (pFile != NULL){


        for (int i = 0; i < input_length; i++){

            fprintf(pFile,"%f\n", input_values[i]);

            }

        for (int i = 0; i < input_length; i++){

            fprintf(pFile,"%f\n", output_values[i]);

            }

        }

	else{

		printf("Could not open the file.\n");
	}


}

	int main () {

    /* Make room for the file with the results */

    remove("result.txt");

    /* Point to text data we want to use  */

	char c1[50] = "NN.txt"; // read text file with network parameters
	char c2[50] = "input.txt"; // read text file with input values (floats, tab separated)

    /* Read the four main parameters of the LSTM network  */

	double parameters[4];
    read_parameters(c1, parameters);
	int timesteps = parameters[0];
	int no_of_batches = parameters[1];
	int no_of_layers = parameters[2];
	int no_of_units = parameters[3];

	/* Read the weights of LSTM network and save them in a structure NN_model  */

    struct LSTM_parameters NN_model;
	read_matrices(c1, &NN_model, no_of_units);

	/* Define the rest of the variables */

    int count = 0; // counter to terminate computation
	double result; //intermidiate result from the neural network, our y value
    double observation; //single observation added to the batch
    int input_length; // number of all the values passed in input.txt file
    double *sequence = (double *)calloc(timesteps, timesteps * sizeof(double)); // dummy variable used for implementation
    double *single_batch = (double *)calloc(timesteps, timesteps * sizeof(double)); // one batch of observations


    input_length = count_input(c2); // save all values from input.txt into an array of an appropriate size
    double *input_sequence = (double *)calloc(input_length, input_length * sizeof(double));
    double *total_output = (double *)calloc(input_length, input_length * sizeof(double));
    input_to_array(c2, input_length, input_sequence);

    while (count < input_length ){ // iterate over batches

        observation = input_sequence[count];
        input_value_to_vector(observation, timesteps, single_batch, sequence); // append a single observation to the batch
        printf("The current batch (input to the network) is: \n\n");
        print_list(timesteps, single_batch);
        result = LSTM_algorithm_three_LSTM_layers (single_batch, parameters, NN_model); // run the implementation
        printf("The intermidiate result is:     %f\n", result);
        total_output[count] = result;
        count = count+1;
}

append_to_file(input_length, input_sequence, total_output);




	return 0;
    //clock_t begin = clock();
    //clock_t end = clock();
    //double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    //printf("Time elapsed: %f ", time_spent);
}
