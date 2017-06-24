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
#include <algorithm>
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

int K;
int L;
int N;

RealMatrix symmetric_key;

struct StateTPM
{
	enum StateModel{
		OUTPUT_EQUAL,
		OUTPUT_NEQUAL = 1
	};
};

/////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
shark::blas::matrix_transpose<T const> transpose_matrix(shark::blas::matrix_expression<T> const& m)
{
	return shark::blas::matrix_transpose<T const>(m());
}

template<class T>
shark::blas::matrix_expression<T> retMat(shark::blas::matrix_expression<T> mat)
{
	return shark::blas::matrix_expression<T>(mat());
}

template<typename T>
void printBatches(Data<T>& dat)
{
	cout << dec << endl;
	for (auto pos : dat.elements())
		cout << pos;
}

void checkZeroMatrix(RealMatrix& matrix)
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

bool comapreMatrices(RealMatrix& a, RealMatrix& b)
{
	//cout << "a:.... " << a << endl;
	//cout << "b:.... " << b << endl;
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

int RandomBit() 
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

class Ann_Input
{
	Data<RealVector> data;
	independent_bits_engine<mt19937, 128, cpp_int> gen128;
	independent_bits_engine<mt19937, 256, cpp_int> gen256;
	vector<RealVector> local_input;
public:
	Ann_Input() :local_input(K, RealVector(N))
	{
		srand(time(NULL));
		PsaudoRandomGenerate();
		data = createDataFromRange(local_input);
	}
	void generate_128bit();
	void generate_256bit();
	void PsaudoRandomGenerate();
	void bitsetRandomPush(deque<std::bitset<64>>& arr);
	Data<RealVector> getBatch(){ return data; }
	std::vector<RealVector> getInput() { return local_input; }
	void renderInput()
	{
		local_input.clear();
		local_input.resize(K, RealVector(N));
		PsaudoRandomGenerate();
		data = createDataFromRange(local_input);
	}
	friend ostream& operator << (ostream& os, Ann_Input& ann);
};

///////////// Ann_Input member
Ann_Input* input;
////////////////

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
	for (int i = 0; i < N * K; i += N)
	{
		RealVector vec;
		for (int j = 0; j < N; j++)
			vec.push_back(RandomBit());
		local_input[k++] = vec;
	}
}


void Ann_Input::bitsetRandomPush(deque<std::bitset<64>>& arr)
{
	int k = 0;
	for (deque<std::bitset<64>>::iterator it = arr.begin(); it != arr.end(); it++)
	{
		for (int i = 0; i < it->size(); i += N)
		{
			RealVector bits;
			for (int j = 0; j < N; j++)
			{
				if ((*it)[i + j] == 0)
					bits.push_back(-1);
				else
					bits.push_back((*it)[i + j]);
			}
			local_input[k++] = bits;
		}
	}
}

void Ann_Input::generate_128bit()
{
	int128_t vec = gen128().convert_to<int128_t>();
	cout << hex << showbase;
	cout << "Generate random input: " << vec << endl;
	cout << "............" << endl << "................." << endl;
	deque<std::bitset<64>> arr;
	arr.push_front(std::bitset<64>(static_cast<uint64_t>(vec)));
	arr.push_front(std::bitset<64>(static_cast<uint64_t>(vec >> 64)));
	bitsetRandomPush(arr);
}

void Ann_Input::generate_256bit()
{
	int256_t vec = gen256().convert_to<int256_t>();
	cout << hex << showbase;
	cout << "Generate random input: " << vec << endl;
	cout << "............" << endl << "................." << endl;
	deque<std::bitset<64>> arr;
	arr.push_front(std::bitset<64> (static_cast<uint64_t>(vec)));
	arr.push_front(std::bitset<64> (static_cast<uint64_t>(vec >> 64)));
	arr.push_front(std::bitset<64> (static_cast<uint64_t>(vec >> 128)));
	arr.push_front(std::bitset<64> (static_cast<uint64_t>(vec >> 192)));
	bitsetRandomPush(arr);
}


/////////////////////////////////////////////////////////////////////////////////////////

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

template<class H, class I, class O>
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
	Model_Ann();
	Model_Ann(std::vector<int> parms, int lim);
	virtual ~Model_Ann() { }
	FFNet<H, O>* getModel() const { return network; }
	void HebbianFunction(vector<int> elements);
	int evalLayerModel();
	void UpdateWeightVector();
	void diagonal_hidden_layer(RealMatrix& matrix, RealMatrix& hidden_mat);
	RealMatrix getParameterWeight()
	{
		symmetric_key = network->layerMatrix(0);
		return network->layerMatrix(0);
	}
};

template<class H, class I, class O>
Model_Ann<H, I, O>::Model_Ann(std::vector<int> parms, int lim) :network(new FFNet<H, O>()), output(1)
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

template<class H, class I, class O>
Model_Ann<H, I, O>::Model_Ann() :numInput(N), numHidden(K), numOutput(1), limit(L), network(new FFNet<H, O>()), output(1)
{
	cout << "Building Model" << endl;
	cout << "	" << N << " input neurons" << endl;
	cout << "	" << K << " hidden neurons" << endl;
	cout << "	1 output neurons" << endl;
	cout << "   limit [" << L << ", -" << L << "]" << endl;
	network->setStructure(numInput, numHidden, numOutput, FFNetStructures::Normal, false);
	initRandomUniform(*network, (int)limit*(-1), limit);
}

template<class H, class I, class O>
void Model_Ann<H, I, O>::HebbianFunction(vector<int> elements)
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


template<class H, class I, class O>
int Model_Ann<H, I, O>::evalLayerModel()
{
	output = 1;
	RealMatrix outputmat;
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

template<class H, class I, class O>
void Model_Ann<H, I, O>::UpdateWeightVector()
{
	vector<int> update_vector;
	for (int i = 0; i < diag_hidden_layer.size(); i++)
	{
		if (output == diag_hidden_layer[i])
			update_vector.push_back(i);
	}
	HebbianFunction(update_vector);
}


template<class H, class I, class O>
void Model_Ann<H, I, O>::diagonal_hidden_layer(RealMatrix& matrix, RealMatrix& hidden_mat)
{
	diag_hidden_layer.clear();
	for (int i = 0; i < matrix.size1(); i++)
	{
		diag_hidden_layer.push_back(matrix(i, i));
		hidden_mat(0, i) = matrix(i, i);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////

class TPM
{
private:
	typedef Model_Ann<SignNeuron, RealVector, LinearNeuron> Model;
	Model model;
	RealMatrix key;
	string id;
	int output;
public:
	TPM(string name) : id(name), output(1) {}
	Model TPMgetModel() const { return model; }
	int getOutput() const { return output; }
	void runModel(char choose);
	void run(TPM& machine);
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
		key = model.getParameterWeight();
		break;
	default:
		break;
	}
}


void simulation(TPM& a, TPM& b)
{
	StateTPM::StateModel state = StateTPM::OUTPUT_NEQUAL;
	while (state != StateTPM::OUTPUT_EQUAL)
	{
		if (comapreMatrices(a.TPMgetModel().getParameterWeight(), b.TPMgetModel().getParameterWeight()))
		{
			cout << "Success to synchronized two models" << endl;
			cout << symmetric_key << endl;
			int *arr= new int[symmetric_key.size1()*symmetric_key.size2()];
			std::ostream_iterator<int> out_it(std::cout, ", ");
			for (int i = 0; i < symmetric_key.size1(); i++)
			{
				for (int j = 0; j < symmetric_key.size2(); j++)
					arr[i*symmetric_key.size2() + j] = symmetric_key(i, j);
			}

			string theString;
			stringstream strStream(stringstream::in | stringstream::out);
			for (int i = 0; i < symmetric_key.size1()*symmetric_key.size2(); i++)
				strStream << arr[i];
			theString = strStream.str();
			CryptoPP::SHA1 sha1;
			string hash = "";
			CryptoPP::StringSource(theString, true, new CryptoPP::HashFilter(sha1, new CryptoPP::HexEncoder(new CryptoPP::StringSink(hash))));
			cout <<"hash is: " << hash << endl;
			state = StateTPM::OUTPUT_EQUAL;
		}
		else
		{
			a.runModel('r');
			b.runModel('r');
			if (a.getOutput() == b.getOutput())
			{
				a.runModel('u');
				b.runModel('u');
			}
			input->renderInput();
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{ 
	switch (argc)
	{
	case 1:
		K = 10;
		L = 3;
		N = 3;
		break;
	case 4:
		K = atoi(argv[1]);
		L = atoi(argv[2]);
		N = atoi(argv[3]);
		break;
	default:
		exit(1);
	}
	cout << "r- Run simulation, q- Quit" << endl;
	char c;
	cin >> c;
	while (c != 'q')
	{
		input = new Ann_Input();
		TPM a("a");
		TPM b("b");
		simulation(a, b);
		cin >> c;
	}
	getchar();
	return 0;
}
