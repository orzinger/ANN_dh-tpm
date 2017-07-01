#pragma once

#include <shark/Core/Shark.h>
#include <shark/Models/FFNet.h>
#include <iostream>
#include <boost\multiprecision\cpp_int.hpp>
#include <bitset>
#include <boost\random.hpp>
#include <boost\random\mersenne_twister.hpp>
#include <boost\serialization\vector.hpp>
#include <boost\math\special_functions\sign.hpp>
#include <shark\Data\BatchInterface.h>
#include <boost\numeric\ublas\operation.hpp>
#include <thread>
#include <time.h>
#include <functional>
#include "..\cryptopp\hex.h"
#include "..\cryptopp\filters.h"
#include "..\cryptopp\sha.h"


#define SL 1 // second layer
#define FL 0 // first layer

using namespace shark;
using namespace std;
using namespace boost::multiprecision;
using namespace boost::random;
using namespace boost::interprocess;


///Global parameters K, L, N
static struct Parms
{
	int K;
	int L;
	int N;
} parms;

///Final private key for both sided
RealMatrix symmetric_key;

///Specifies whether print weights vector or not
uint16_t debug=0;

//Specifies whether A and B weights vectors are equal
struct StateTPM
{
	enum StateModel{
		OUTPUT_EQUAL,
		OUTPUT_NEQUAL = 1
	};
};



/////////////////////////////////////////////////////////////////////////////////////////////
/// Give transpose matrix
template<class T>
shark::blas::matrix_transpose<T const> transpose_matrix(shark::blas::matrix_expression<T> const& m)
{
	return shark::blas::matrix_transpose<T const>(m());
}
///Return matrix as matrix expression
template<class T>
shark::blas::matrix_expression<T> retMat(shark::blas::matrix_expression<T> mat)
{
	return shark::blas::matrix_expression<T>(mat());
}
///Print Xij of input data
template<typename T>
void printBatches(Data<T>& dat)
{
	cout << dec << endl;
	for (auto pos : dat.elements())
		cout << pos;
}
/// Check if there are zeros as result of eval layer
inline void checkZeroMatrix(RealMatrix& matrix)
{
	for (int i = 0; i < matrix.size1(); i++)
	{
		for (int j = 0; j < matrix.size2(); j++)
		{
			if (matrix(i, j) == 0)
				matrix(i, j) = -1;
		}
	}
}
/// Compare matrices
inline bool comapreMatrices(RealMatrix& a, RealMatrix& b)
{
	if (debug)
	{
		cout << "a:.... " << a << endl;
		cout << "b:.... " << b << endl;
	}
	if ((a.size1() != b.size1()) && (a.size2() != b.size2()))
		return false;
	for (int i = 0; i < a.size1(); i++)
	{
		for (int j = 0; j < a.size2(); j++)
		{
			if (a(i, j) != b(i, j))
				return false;
		}
	}
	return true;
}
/// Randomize a bit: -1 or 1
inline int RandomBit()
{
	int result, A;
	A = rand() % 2;
	if (A == 0)
		result = -1;
	else
		result = 1;
	return result;
}


////////////////////////////////////////////////////////////////////////////////////////
///Class Ann_Input represnt a class handle
///Input data, synthesizes it and make it input data
///to unsupervised learning model. This is a public access
///Params:
///	_data- container with batches of data
///	local_input- an array of bits to to _data
/// gen128- future use for auto generate 128bit numbers
/// gen256- future use for auto generate 256bit numbers
///
class Ann_Input
{
	Data<RealVector> _data;
	independent_bits_engine<mt19937, 128, cpp_int> gen128;
	independent_bits_engine<mt19937, 256, cpp_int> gen256;
	vector<RealVector> local_input;
public:
	Ann_Input() :local_input(parms.K, RealVector(parms.N))
	{
		srand(time(NULL));
		PsaudoRandomGenerate();
		_data = createDataFromRange(local_input);
	}
	///Generate a K*N bits [-1,1] of input data
	void PsaudoRandomGenerate();
	///Return _data
	Data<RealVector> getBatch(){ return _data; }
	///Return local_input
	std::vector<RealVector> getInput() { return local_input; }
	///Render an input
	void renderInput()
	{
		local_input.clear();
		local_input.resize(parms.K, RealVector(parms.N));
		PsaudoRandomGenerate();
		_data = createDataFromRange(local_input);
	}
	///Operator << for print object of class
	friend ostream& operator << (ostream& os, Ann_Input& ann);
};

ostream& operator << (ostream& os, Ann_Input& ann)
{
	std::vector<RealVector> v = ann.getInput();
	std::cout << dec << endl;
	for (auto it = v.begin(); it != v.end(); it++)
	{
		for (auto it2 = it->begin(); it2 != it->end(); it2++)
			os << *it2;
	}
	std::cout << endl;
	return os;
}

void Ann_Input::PsaudoRandomGenerate()
{
	int k = 0;
	for (int i = 0; i < parms.N * parms.K; i += parms.N)
	{
		RealVector vec;
		for (int j = 0; j < parms.N; j++)
			vec.push_back(RandomBit());
		local_input[k++] = vec;
	}
}

/////////////
///A global Ann_Input pointer
///for Model_Ann
Ann_Input* input;



/////////////////////////////////////////////////////////////////////////////////////////
///Activation function, implement Sign funcion, for our network
struct SignNeuron : public shark::detail::NeuronBase<SignNeuron>
{
	template<class T>
	T function(T x) const
	{
		return boost::math::sign(x);
	}
	template<class T>
	T functionDerivative(T x) const
	{
		return 0;
	}
};

///Class Model_Ann represent a class handle with
///modeling, functionality and synchronization of the neural network
///Construct as a template for various executable functions
///H- Hidden neurons activation function
///O- Output neuron actvation function
///Parms:
///	numInput- number of input neurons
/// numHidden- number of hidden neurons
/// numOutput- number of output neurons (default as 1 always)
/// limit- limit of weights synapses
/// network- model as object of Feed-Forward-Neural-Net
/// diag_hidden_layer- array with result of hidden layer
/// output- output of the model
template<class H, class O>
class Model_Ann
{
	unsigned int numInput;
	unsigned int numHidden;
	unsigned int numOutput;
	unsigned int limit;
	FFNet<H, O>* network;
	vector<int> diag_hidden_layer;
	int output;
public:
	//Default Ctor
	///take K,L and N as parameters
	Model_Ann();
	///Ctor
	///take user parameters if he wants to control himself on the ctor 
	Model_Ann(std::vector<int> parms, int lim);
	///Virtual Dtor
	virtual ~Model_Ann() { }
	///Return model
	FFNet<H, O>* getModel() const { return network; }
	///Random walk learning algorithm
	void Random_Walk(vector<int> elements);
	///Evaluate the hidden layer
	int evalLayerModel();
	///Update the weights vector
	void UpdateWeightVector();
	///Update the hidden neurons outputs
	void diagonal_hidden_layer(RealMatrix& matrix, RealMatrix& hidden_mat);
	///Return the weights vector
	RealMatrix getParameterWeight()
	{
		symmetric_key = network->layerMatrix(0);
		return network->layerMatrix(0);
	}
};

template<class H, class O>
Model_Ann<H, O>::Model_Ann(std::vector<int> parms, int lim) :network(new FFNet<H, O>()), output(1)
{
	numInput = parms[0];
	numHidden = parms[1];
	numOutput = parms[2];
	limit = lim;
	cout << "Building Model" << endl;
	cout << "	" << numInput << " input neurons" << endl;
	cout << "	" << numHidden << " hidden neurons" << endl;
	cout << "	" << numOutput << " output neurons" << endl;
	network->setStructure(numInput, numHidden, numOutput, FFNetStructures::Normal, false);
	initRandomUniform(*network, (int)limit*(-1), limit);
}

template<class H, class O>
Model_Ann<H, O>::Model_Ann() :numInput(parms.N), numHidden(parms.K), numOutput(1), limit(parms.L), network(new FFNet<H, O>()), output(1)
{
	cout << "Building Model" << endl;
	cout << "	" << parms.N << " input neurons" << endl;
	cout << "	" << parms.K << " hidden neurons" << endl;
	cout << "	1 output neurons" << endl;
	cout << "   limit [" << parms.L << ", -" << parms.L << "]" << endl;
	network->setStructure(numInput, numHidden, numOutput, FFNetStructures::Normal, false);
	initRandomUniform(*network, (int)limit*(-1), limit);
}

template<class H, class O>
void Model_Ann<H, O>::Random_Walk(vector<int> elements)
{
	unique_ptr<RealMatrix> temp_mat(new RealMatrix(network->layerMatrix(0)));
	for (vector<int>::iterator element_pos = elements.begin(); element_pos != elements.end(); element_pos++)
	{
		for (int j = 0; j < temp_mat->size2(); j++)
		{
			int x = (int)((int)(network->layerMatrix(0)(*element_pos, j)) + (int)(input->getBatch().element(*element_pos)(j)));
			if (x > limit)
				x = limit;
			if (x < (int)limit*(-1))
				x = (int)limit*(-1);
			(*temp_mat)(*element_pos, j) = x;
		}
	}
	network->setLayer(0, *temp_mat);
}


template<class H, class O>
int Model_Ann<H, O>::evalLayerModel()
{
	output = 1;
	RealMatrix outputmat;
	//Calculate hidden layer, with the input batcher Xij
	network->evalLayer(0, input->getBatch().batch(0), outputmat);
	checkZeroMatrix(outputmat);
	//////////This part update second layer with his output
	RealMatrix hidden_matrix(network->layerMatrix(SL).size1(), network->layerMatrix(SL).size2());
	diagonal_hidden_layer(outputmat, hidden_matrix);
	network->setLayer(SL, hidden_matrix);
	//////////
	for (int i = 0; i < diag_hidden_layer.size(); i++)
		output *= diag_hidden_layer[i];
	return output;
}

template<class H, class O>
void Model_Ann<H, O>::UpdateWeightVector()
{
	vector<int> update_vector;
	for (int i = 0; i < diag_hidden_layer.size(); i++)
	{
		//Only h(i) (hidden layers) that equal to machine's output are pushed
		if (output == diag_hidden_layer[i])
			update_vector.push_back(i);
	}
	Random_Walk(update_vector);
}


template<class H, class O>
void Model_Ann<H, O>::diagonal_hidden_layer(RealMatrix& matrix, RealMatrix& hidden_mat)
{
	///Only the hidden neurons results pushed
	diag_hidden_layer.clear();
	for (int i = 0; i < matrix.size1(); i++)
	{
		diag_hidden_layer.push_back(matrix(i, i));
		hidden_mat(0, i) = matrix(i, i);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////
///Class TPM represent Tree Parity Machine
///Parms
///	model- object of Model_Ann
///	key- Weights vector of TPM
/// id- id of machine
/// output- output of machine
class TPM
{
private:
	typedef Model_Ann<SignNeuron, LinearNeuron> Model;
	Model model;
	string id;
	int output;
public:
	///Ctor
	TPM(string name) : id(name), output(1) {}
	///Return model
	Model TPMgetModel() const { return model; }
	///Return output
	int getOutput() const { return output; }
	///Executing the model
	void runModel(char choose);
	//Return id of machine
	string getID() { return id; }
};

void TPM::runModel(char choose)
{
	switch (choose)
	{
	case 'r':
		output = model.evalLayerModel();
		break;
	case 'u':
		model.UpdateWeightVector();
		break;
	default:
		break;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////




