#include <shark\Ann\Ann.hpp>

static void simulation(TPM& a, TPM& b)
{
	StateTPM::StateModel state = StateTPM::OUTPUT_NEQUAL;
	while (state != StateTPM::OUTPUT_EQUAL)
	{
		///Check if synchronization completed
		if (comapreMatrices(a.TPMgetModel().getParameterWeight(), b.TPMgetModel().getParameterWeight()))
		{
			std::cout << "Success to synchronized two models" << endl;
			std::cout << symmetric_key << endl;
			int *arr = new int[symmetric_key.size1()*symmetric_key.size2()];
			std::ostream_iterator<int> out_it(std::cout, ", ");
			for (int i = 0; i < symmetric_key.size1(); i++)
			{
				for (int j = 0; j < symmetric_key.size2(); j++)
					arr[i*symmetric_key.size2() + j] = symmetric_key(i, j);
			}

			///Generate a hash value to represent the key in hexadecimal format
			///generate it as SHA1 hash value consists of 160 bit
			string theString;
			stringstream strStream(stringstream::in | stringstream::out);
			for (int i = 0; i < symmetric_key.size1()*symmetric_key.size2(); i++)
				strStream << arr[i];
			theString = strStream.str();
			CryptoPP::SHA1 sha1;
			string hash = "";
			CryptoPP::StringSource(theString, true, new CryptoPP::HashFilter(sha1, new CryptoPP::HexEncoder(new CryptoPP::StringSink(hash))));
			std::cout << "hash is: " << hash << endl;
			state = StateTPM::OUTPUT_EQUAL;
		}
		else
		{
			///Run models of A and B
			a.runModel('r');
			b.runModel('r');
			///If outputs of A and B are equal, update weights according to their learning rule
			if (a.getOutput() == b.getOutput())
			{
				a.runModel('u');
				b.runModel('u');
			}
			//Rendering a new input
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
		parms.K = 10;
		parms.L = 3;
		parms.N = 3;
		break;
	case 2:
		if (!strcmp(argv[1], "-h"))
		{
			std::cout << "Usage " << argv[0] << endl;
			std::cout << "-h- Help" << endl;
			std::cout << "User choise- <#hidden> <#limit> <#input> [-debug]" << endl;
			std::cout << "Default- K=10, L=3, N=3 [-debug]" << endl;
			getchar();
			return 1;
		}
		else
		{
			parms.K = 10;
			parms.L = 3;
			parms.N = 3;
			if (!strcmp(argv[1], "-debug"))
				debug = 1;
		}
		break;
	case 4:
		parms.K = atoi(argv[1]);
		parms.L = atoi(argv[2]);
		parms.N = atoi(argv[3]);
		break;
	case 5:
		parms.K = atoi(argv[1]);
		parms.L = atoi(argv[2]);
		parms.N = atoi(argv[3]);
		assert(!strcmp(argv[1], "-h"));
		debug = 1;
		break;
	default:
		exit(1);
	}
	std::cout << "r- Run simulation, q- Quit" << endl;
	char c;
	std::cin >> c;
	while (c != 'q')
	{
		if (c == 'r')
		{
			input = new Ann_Input();
			TPM a("a");
			TPM b("b");
			simulation(a, b);
			delete input;
		}
		std::cin >> c;
	}
	getchar();
	return 0;
}
