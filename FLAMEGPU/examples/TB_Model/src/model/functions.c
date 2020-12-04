#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "header.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <fstream>

#ifdef WIN32
    #include <direct.h>
    #define GetCurrentDir _getcwd
#else
    #include <unistd.h>
    #define GetCurrentDir getcwd
#endif

// Uncomment to enable profiling logging in the console
//#define INSTRUMENT_INIT_FUNCTIONS 1 // Initialisation time
//#define INSTRUMENT_AGENT_FUNCTIONS 1 // Time for each agent function
//#define INSTRUMENT_ITERATIONS 1 // Time for each iteration

#define MEGACHURCH_TEST 1
#define OUTPUT_POPULATION_PER_ITERATION 0

// HOUSEHOLD = 0, CHURCH = 1 etc.
enum Location { HOUSEHOLD, CHURCH, TRANSPORT, CLINIC, WORKPLACE, BAR, SCHOOL, OUTSIDE };

// Allocate blocks of memory for each type of agent, and defining a constant
// for the maximum number of each that should be generated.
xmachine_memory_Person **h_agent_AoS;
xmachine_memory_Person **h_person_AoS;
xmachine_memory_Household **h_household_AoS;
xmachine_memory_Church **h_church_AoS;
xmachine_memory_Transport **h_transport_AoS;
xmachine_memory_Clinic **h_clinic_AoS;
xmachine_memory_Workplace **h_workplace_AoS;
xmachine_memory_Bar **h_bar_AoS;
xmachine_memory_School **h_school_AoS;

xmachine_memory_TBAssignment **h_tbassignment_AoS;
xmachine_memory_HouseholdMembership **h_hhmembership_AoS;
xmachine_memory_ChurchMembership **h_chumembership_AoS;

const unsigned int h_agent_AoS_MAX = 32768;
const unsigned int h_household_AoS_MAX = 8192;
const unsigned int h_church_AoS_MAX = 256;
const unsigned int h_transport_AoS_MAX = 2048;
const unsigned int h_clinic_AoS_MAX = 2;
const unsigned int h_workplace_AoS_MAX = 8192;
const unsigned int h_bar_AoS_MAX = 4096;
const unsigned int h_school_AoS_MAX = 2048;

const unsigned int h_tbassignment_AoS_MAX = 32768;
const unsigned int h_hhmembership_AoS_MAX = 32768;
const unsigned int h_chumembership_AoS_MAX = 8192;
const unsigned int h_trmembership_AoS_MAX = 32768;
const unsigned int h_wpmembership_AoS_MAX = 32768;
const unsigned int h_schmembership_AoS_MAX = 16384;

// Create variables for the next unused ID for each agent type, so that they
// remain unique, and also get functions to update the ID each time.
unsigned int h_nextID = 0;
unsigned int h_nextHouseholdID = 0;
unsigned int h_nextChurchID = 0;
unsigned int h_nextTransportID = 0;
unsigned int h_nextClinicID = 0;
unsigned int h_nextWorkplaceID = 0;
unsigned int h_nextBarID = 0;
unsigned int h_nextSchoolID = 0;

unsigned int personCounter = 0;

__host__ unsigned int getNextID()
{
  unsigned int old = h_nextID;
  h_nextID++;
  return old;
}

__host__ unsigned int getNextHouseholdID()
{
  unsigned int old = h_nextHouseholdID;
  h_nextHouseholdID++;
  return old;
}

__host__ unsigned int getNextChurchID()
{
  unsigned int old = h_nextChurchID;
  h_nextChurchID++;
  return old;
}

__host__ unsigned int getNextTransportID()
{
  unsigned int old = h_nextTransportID;
  h_nextTransportID++;
  return old;
}

__host__ unsigned int getNextWorkplaceID()
{
  unsigned int old = h_nextWorkplaceID;
  h_nextWorkplaceID++;
  return old;
}

__host__ unsigned int getNextBarID()
{
  unsigned int old = h_nextBarID;
  h_nextBarID++;
  return old;
}

__host__ unsigned int getNextSchoolID()
{
  unsigned int old = h_nextSchoolID;
  h_nextSchoolID++;
  return old;
}

// A function to shuffle two arrays with the same permutation, so that
// information can be kept together during the shuffle.
__host__ void shuffle(unsigned int *array1, unsigned int *array2, size_t n)
{
  if (n > 1)
  {
    size_t i;
    for (i = 0; i < n - 1; i++)
    {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      unsigned int t1 = array1[j];
      unsigned int t2 = array2[j];
      array1[j] = array1[i];
      array2[j] = array2[i];
      array1[i] = t1;
      array2[i] = t2;
    }
  }
}

void swap(unsigned int *a, unsigned int *b)
{
  unsigned int t = *a;
  *a = *b;
  *b = t;
}

unsigned int partition(unsigned int *arr1, unsigned int *arr2, unsigned int low,
                       unsigned int high)
{
  unsigned int pivot = arr1[high]; // pivot
  unsigned int i = (low - 1);      // Index of smaller element

  for (unsigned int j = low; j <= high - 1; j++)
  {
    // If current element is smaller than or
    // equal to pivot
    if (arr1[j] <= pivot)
    {
      i++; // increment index of smaller element
      swap(&arr1[i], &arr1[j]);
      swap(&arr2[i], &arr2[j]);
    }
  }
  swap(&arr1[i + 1], &arr1[high]);
  swap(&arr2[i + 1], &arr2[high]);
  return (i + 1);
}

void quickSort(unsigned int *arr1, unsigned int *arr2, unsigned int low,
               unsigned int high)
{
  if (low < high)
  {
    /* pi is partitioning index, arr[p] is now
           at right place */
    unsigned int pi = partition(arr1, arr2, low, high);

    // Separately sort elements before
    // partition and after partition
    if (pi != 0)
    {
      quickSort(arr1, arr2, low, pi - 1);
      quickSort(arr1, arr2, pi + 1, high);
    }
    else
    {
      quickSort(arr1, arr2, pi + 1, high);
    }
  }
}

// A function that returns the day of the week given an iteration number of
// increments of 5 minutes, in the form Sunday = 0, Monday = 1 etc.
__device__ unsigned int dayofweek(unsigned int step)
{
  return (step % 2016) / 288;
}

__device__ unsigned int dayofmonth(unsigned int step)
{
  return (step % 8064) / 288;
}

//__device__ float cadgamma(float kk, float u, float v, float w)
__device__ float cadgamma(float kk, float u, float v, float w, const int limit = 1000)
{
  float xi = 1.0;
  float eta = 1.0;
  int ct = 0;

  while (eta > pow(xi, (kk - 1)) * exp(-xi))
  {
	  ct++;
    if (u * (E + kk) < E)
    {
      xi = pow(v, 1 / kk);
      eta = w * v / xi;
    }
    else
    {
      xi = 1 - log(v);
      eta = w * v / E;
    }
	if (ct > limit) break;
  }

  return xi;
}

// A struct to represent a time of day, and a function that returns a time of
// day given an iteration number of increments of 5 minutes.
struct Time
{
  unsigned int hour;
  unsigned int minute;
};

__device__ struct Time timeofday(unsigned int step)
{
  unsigned int hour = (step % 288) / 12;
  unsigned int minute = (step % 12) * 5;
  Time t = {hour, minute};
  return t;
}

// Used for networked messaging - essentially provides a unique index for each location to read/write messages to/from
__device__ unsigned int locationToEdgeID(unsigned int location, unsigned int locationID) {
	return LOCATION_EDGE_OFFSETS[location] + locationID;
}

__device__ float device_exp(float x)
{
  float y = exp(x);
  return y;
}

typedef struct AgeGenderCategory {
	unsigned int gender;
	unsigned int minage;
	unsigned int maxage;
	unsigned int* numPeopleInHouseholdSize;
} AgeGenderCategory;

typedef struct PopulationInfo {
	unsigned int totalPopulationSize;
	unsigned int numHouseholdSizes;
	unsigned int numAgeGenderCategories;
	unsigned int numAdults;
	unsigned int* householdSizes;
	unsigned int* numPeopleInHouseholdSize;
	AgeGenderCategory* ageGenderCategories;
} PopulationInfo;


// Holder for all the arrays used during initialisation - in future it would be good to refactor these so they are local to where they're used
typedef struct DataHolder {
	unsigned int ages[h_agent_AoS_MAX];
	unsigned int daycount = 0;
	unsigned int transport[h_agent_AoS_MAX];
	unsigned int days[h_agent_AoS_MAX];
	unsigned int tbarray[h_agent_AoS_MAX];
	float weights[h_agent_AoS_MAX];
	unsigned int employedcount = 0;
	unsigned int workarray[h_agent_AoS_MAX];
	unsigned int barcount = 0;
	unsigned int childcount = 0;
	unsigned int schoolarray[h_agent_AoS_MAX];

	// Create an array to keep track of how many adults are in each household, for
	// use when generating churches.
	unsigned int adult[h_household_AoS_MAX];
	unsigned int adultcount;

	unsigned int *order;

	unsigned int total;

	unsigned int activehouseholds[h_household_AoS_MAX];
	unsigned int* activepeople;

	// Create an array to store the number of people who live in households of
 // each size, and start each size at 0 people; also create an array of the
 // ages of people with given ids, used for church generation.
	unsigned int sizesarray[h_agent_AoS_MAX];

	signed int count;
} DataHolder;

// Reads the data file and stores all the information 
PopulationInfo readInputFile() {
	PopulationInfo populationInfo;

	// Initialise the input file generated by Python preprocessing, from which
	// the data about people will be read in.
	
#ifdef WIN32
        char const* const fileName = "input/data.in"; // "../../../examples/TB_Model/iterations/data.in"; //"data.in"
#else
        char const* const fileName = "input/data.in"; // "../../../examples/TB_Model/iterations/data.in"; //"data.in"
#endif
	FILE *file = fopen(fileName, "r");
	char line[256];

	char buff[FILENAME_MAX];
	GetCurrentDir(buff, FILENAME_MAX);
	printf("Current working dir: %s\n", buff);

	if (file == NULL) {
		printf("Coudn't open file");
	} else {
	       printf("File opened successfully\n");
	}
	

	// Read in the total number of people, household sizes and age/gender
	// categories, as these will be how many times we need to loop.
	populationInfo.totalPopulationSize = strtol(fgets(line, sizeof(line), file), NULL, 0);
	populationInfo.numHouseholdSizes = strtol(fgets(line, sizeof(line), file), NULL, 0);
	populationInfo.numAgeGenderCategories = strtol(fgets(line, sizeof(line), file), NULL, 0);
	populationInfo.numAdults = strtol(fgets(line, sizeof(line), file), NULL, 0);

	populationInfo.householdSizes = (unsigned int*)malloc(sizeof(int)*populationInfo.numHouseholdSizes);
	populationInfo.numPeopleInHouseholdSize = (unsigned int*)malloc(sizeof(unsigned int)*populationInfo.numHouseholdSizes);
        for (unsigned int i = 0; i < populationInfo.numHouseholdSizes; i++) {
            populationInfo.numPeopleInHouseholdSize[i] = 0;
        }
	populationInfo.ageGenderCategories = (AgeGenderCategory*)malloc(sizeof(AgeGenderCategory)*populationInfo.numAgeGenderCategories);

	for (int i = 0; i < populationInfo.numAgeGenderCategories; i++) {
		// Read in the age/gender category we are working with. (Max age can be
		// infinity, in which case we set this to the max age parameter given in the
		// input.)
		populationInfo.ageGenderCategories[i].gender = strtol(fgets(line, sizeof(line), file), NULL, 0);
		populationInfo.ageGenderCategories[i].minage = strtol(fgets(line, sizeof(line), file), NULL, 0);

		// Add one to minage unless minage is 0? Why? TODO: Check with Pete Dodd if this is correct
		if ((int)populationInfo.ageGenderCategories[i].minage != 0)
		{
			populationInfo.ageGenderCategories[i].minage++;
		}

		// Set max age for the ageGenderCategory
		char* maxagestring = fgets(line, sizeof(line), file);

		if (!strcmp(maxagestring, "Inf\n"))
		{
			populationInfo.ageGenderCategories[i].maxage = (*get_MAX_AGE());
		}
		else
		{
			populationInfo.ageGenderCategories[i].maxage = strtol(maxagestring, NULL, 0);
		}

		populationInfo.ageGenderCategories[i].numPeopleInHouseholdSize = (unsigned int*)malloc(sizeof(unsigned int*)*populationInfo.numHouseholdSizes);
		for (int j = 0; j < populationInfo.numHouseholdSizes; j++) {
			// Read in the household size we are working with and the relevant value
			// from the histogram.
			populationInfo.householdSizes[j] = strtol(fgets(line, sizeof(line), file), NULL, 0);
			populationInfo.ageGenderCategories[i].numPeopleInHouseholdSize[j] = strtof(fgets(line, sizeof(line), file), NULL);
			populationInfo.numPeopleInHouseholdSize[j] += populationInfo.ageGenderCategories[i].numPeopleInHouseholdSize[j];
		}

	}
	fclose(file);

	return populationInfo;
}

void initAgentTypes() {
	// Initialise all of the agent types with an id of 1 and allocating an array
	// of memory for each one. TODO: Should these be starting from 1 or 0?
	h_nextID = 0;
	h_agent_AoS = h_allocate_agent_Person_array(h_agent_AoS_MAX);
	h_nextHouseholdID = 0;
	h_household_AoS = h_allocate_agent_Household_array(h_household_AoS_MAX);
	h_nextChurchID = 0;
	h_church_AoS = h_allocate_agent_Church_array(h_church_AoS_MAX);
	h_nextTransportID = 0;
	h_transport_AoS = h_allocate_agent_Transport_array(h_transport_AoS_MAX);
	h_nextClinicID = 0;
	h_nextWorkplaceID = 0;
	h_workplace_AoS = h_allocate_agent_Workplace_array(h_workplace_AoS_MAX);
	h_nextBarID = 0;
	h_bar_AoS = h_allocate_agent_Bar_array(h_bar_AoS_MAX);
	h_tbassignment_AoS =
		h_allocate_agent_TBAssignment_array(h_tbassignment_AoS_MAX);
	//h_hhmembership_AoS =
	//	h_allocate_agent_HouseholdMembership_array(h_hhmembership_AoS_MAX);
	h_chumembership_AoS =
		h_allocate_agent_ChurchMembership_array(h_chumembership_AoS_MAX);
}

/* Each of the 'decide' functions have no dependency on any other agent and could be run concurrently in FGPU2. They each use a randomly generated number & set of probabilities to decide the properties of the person.
   For example, their ages, whether they go to bars, how often they use transport. */

void decideAgeHivArt(xmachine_memory_Person *h_person, DataHolder& dh, const PopulationInfo& populationInfo, int minage, int maxage, int gender) {

	float rr_as_f_46 = *get_RR_AS_F_46();
	float rr_as_f_26 = *get_RR_AS_F_26();
	float rr_as_f_18 = *get_RR_AS_F_18();
	float rr_as_m_46 = *get_RR_AS_M_46();
	float rr_as_m_26 = *get_RR_AS_M_26();
	float rr_as_m_18 = *get_RR_AS_M_18();

	float starting_population = *get_STARTING_POPULATION();
	float hiv_prevalence = *get_HIV_PREVALENCE();
	float art_coverage = *get_ART_COVERAGE();
	float rr_hiv = *get_RR_HIV();
	float rr_art = *get_RR_ART();
	float hivprob = hiv_prevalence * starting_population / populationInfo.numAdults;
	
	float random = ((float)rand() / (RAND_MAX));

	float rr_as;
	float weight;

	// Pick a random age for the person between the bounds of the age
				// interval they belong to.
	int age = (rand() % (maxage + 1 - minage)) + minage;
	h_person->age = age;

	// Decide risk based on age & gender
	if (gender == 2)
	{
		if (age >= 46)
		{
			rr_as = rr_as_f_46;
		}
		else if (age >= 26)
		{
			rr_as = rr_as_f_26;
		}
		else if (age >= 18)
		{
			rr_as = rr_as_f_18;
		}
		else
		{
			rr_as = 0.00;
		}
	}
	else
	{
		if (age >= 46)
		{
			rr_as = rr_as_m_46;
		}
		else if (age >= 26)
		{
			rr_as = rr_as_m_26;
		}
		else if (age >= 18)
		{
			rr_as = rr_as_m_18;
		}
		else
		{
			rr_as = 0.00;
		}
	}

	// Decide if user has hiv/art
	random = ((float)rand() / (RAND_MAX));

	if ((random < hivprob) && (h_person->age >= 18))
	{
		h_person->hiv = 1;

		random = ((float)rand() / (RAND_MAX));
		if (random < art_coverage)
		{
			h_person->art = 1;
			weight = rr_as * rr_hiv * rr_art;

			unsigned int randomday = rand() % 28;

			while (randomday % 7 == 0 || randomday % 7 == 6)
			{
				randomday = rand() % 28;
			}

			h_person->artday = randomday;
		}
		else
		{
			h_person->art = 0;
			weight = rr_as * rr_hiv;
		}
	}
	else
	{
		h_person->hiv = 0;
		h_person->art = 0;
		weight = rr_as;
	}

	dh.weights[h_person->id] = weight;

	// Update the arrays of information with this person's household size
	// and age.
	dh.ages[h_person->id] = age;
}

void decideTransport(xmachine_memory_Person *h_person, DataHolder& dh) {
	// Decide whether the person is a transport user based on given input
					// probabilities.
	float transport_beta0 = *get_TRANSPORT_BETA0();
	float transport_beta1 = *get_TRANSPORT_BETA1();
	float transport_freq0 = *get_TRANSPORT_FREQ0();
	float transport_freq2 = *get_TRANSPORT_FREQ2();

	float random = ((float)rand() / (RAND_MAX));
	float useprob =
		1.0 / (1 + exp(-transport_beta0 - (transport_beta1 * h_person->age)));

	unsigned int transportuser;
	if (random < useprob)
	{
		transportuser = 1;
	}
	else
	{
		transportuser = 0;
	}

	// If the person is a transport user, pick a transport frequency and
	// duration for them based on input probabilities; otherwise, set these
	// variables to a dummy value.
	if (transportuser)
	{
		random = ((float)rand() / (RAND_MAX));

		if (random < transport_freq0)
		{
		}
		else if (random < transport_freq2)
		{
			h_person->transportday1 = (rand() % 5) + 1;
			h_person->transportday2 = -1;

			dh.transport[dh.daycount] = h_person->id;
			dh.days[dh.daycount] = h_person->transportday1;

			dh.daycount++;
		}
		else
		{
			h_person->transportday1 = (rand() % 5) + 1;
			h_person->transportday2 = h_person->transportday1;

			while (h_person->transportday2 == h_person->transportday1)
			{
				h_person->transportday2 = (rand() % 5) + 1;
			}

			dh.transport[dh.daycount] = h_person->id;
			dh.days[dh.daycount] = h_person->transportday1;

			dh.daycount++;

			dh.transport[dh.daycount] = h_person->id;
			dh.days[dh.daycount] = h_person->transportday2;

			dh.daycount++;
		}
	}
	else
	{
		h_person->transport = -1;
		h_person->transportday1 = -1;
		h_person->transportday2 = -1;
	}
}

void decideWork(xmachine_memory_Person *h_person, DataHolder& dh) {
	float workplace_beta0 = *get_WORKPLACE_BETA0();
	float workplace_betaa = *get_WORKPLACE_BETAA();
	float workplace_betas = *get_WORKPLACE_BETAS();
	float workplace_betaas = *get_WORKPLACE_BETAAS();

	unsigned int sex = h_person->gender % 2;
	float random = ((float)rand() / (RAND_MAX));
	float workprob =
		1.0 /
		(1 + exp(-workplace_beta0 - (workplace_betaa * h_person->age) -
		(workplace_betas * sex) -
			(workplace_betaas * sex * h_person->age)));

	random = ((float)rand() / (RAND_MAX));

	if (random < workprob && h_person->age >= 18)
	{
		dh.workarray[dh.employedcount] = h_person->id;
		dh.employedcount++;
	}
}

void decideBar(xmachine_memory_Person *h_person, DataHolder& dh) {
	float bar_m_prob1 = *get_BAR_M_PROB1();
	float bar_m_prob2 = *get_BAR_M_PROB2();
	float bar_m_prob3 = *get_BAR_M_PROB3();
	float bar_m_prob4 = *get_BAR_M_PROB4();
	float bar_m_prob5 = *get_BAR_M_PROB5();
	float bar_m_prob7 = *get_BAR_M_PROB7();
	float bar_f_prob1 = *get_BAR_F_PROB1();
	float bar_f_prob2 = *get_BAR_F_PROB2();
	float bar_f_prob3 = *get_BAR_F_PROB3();
	float bar_f_prob4 = *get_BAR_F_PROB4();
	float bar_f_prob5 = *get_BAR_F_PROB5();
	float bar_f_prob7 = *get_BAR_F_PROB7();

	float bar_beta0 = *get_BAR_BETA0();
	float bar_betaa = *get_BAR_BETAA();
	float bar_betas = *get_BAR_BETAS();
	float bar_betaas = *get_BAR_BETAAS();

	unsigned int sex = h_person->gender % 2;
	float barprob =
		1.0 /
		(1 + exp(-bar_beta0 - (bar_betaa * h_person->age) -
		(bar_betas * sex) - (bar_betaas * sex * h_person->age)));

	float random = ((float)rand() / (RAND_MAX));

	if (random < barprob && h_person->age >= 18)
	{
		random = ((float)rand() / (RAND_MAX));
		h_person->bargoing = 1;

		if (sex == 0)
		{
			if (random < bar_f_prob1)
			{
				h_person->barday = 1;
				dh.barcount++;
			}
			else if (random < bar_f_prob2)
			{
				h_person->barday = 2;
				dh.barcount += 2;
			}
			else if (random < bar_f_prob3)
			{
				h_person->barday = 3;
				dh.barcount += 3;
			}
			else if (random < bar_f_prob4)
			{
				h_person->barday = 4;
				dh.barcount += 4;
			}
			else if (random < bar_f_prob5)
			{
				h_person->barday = 5;
				dh.barcount += 5;
			}
			else if (random < bar_f_prob7)
			{
				h_person->barday = 7;
				dh.barcount += 7;
			}
			else
			{
				h_person->bargoing = 2;
				h_person->barday = rand() % 28;
			}
		}
		else // Sex != 0 
		{
			if (random < bar_m_prob1)
			{
				h_person->barday = 1;
				dh.barcount++;
			}
			else if (random < bar_m_prob2)
			{
				h_person->barday = 2;
				dh.barcount += 2;
			}
			else if (random < bar_m_prob3)
			{
				h_person->barday = 3;
				dh.barcount += 3;
			}
			else if (random < bar_m_prob4)
			{
				h_person->barday = 4;
				dh.barcount += 4;
			}
			else if (random < bar_m_prob5)
			{
				h_person->barday = 5;
				dh.barcount += 5;
			}
			else if (random < bar_m_prob7)
			{
				h_person->barday = 7;
				dh.barcount += 7;
			}
			else
			{
				h_person->bargoing = 2;
				h_person->barday = rand() % 28;
			}
		}
	}
	else
	{
		h_person->bargoing = 0;
	}
}

void decideSchool(xmachine_memory_Person *h_person, DataHolder& dh) {
	// If the person is younger than 18, list them as requiring a school
	if (h_person->age < 18)
	{
		dh.schoolarray[dh.childcount] = h_person->id;
		dh.childcount++;
	}
}

/* Init people generates the person agents. */
void initPeople(const PopulationInfo& populationInfo, DataHolder& dh) {
	unsigned int max_age = *get_MAX_AGE();
	unsigned int starting_population = populationInfo.totalPopulationSize;

	// Allocate memory for the agents we are generating.
	h_person_AoS = h_allocate_agent_Person_array((int)(starting_population));

	// This loop runs once for each age/gender category, so once for every row in
	// the histogram.
	for (unsigned int i = 0; i < populationInfo.numAgeGenderCategories; i++)
	{
		int gender = populationInfo.ageGenderCategories[i].gender;
		int maxage = populationInfo.ageGenderCategories[i].maxage;
		int minage = populationInfo.ageGenderCategories[i].minage;

		// This loop runs once for each size of household, so once for every column
		// in the histogram. At this point we are working with individual values
		// from the histogram.
		for (unsigned int j = 0; j < populationInfo.numHouseholdSizes; j++)
		{
			int currentsize = populationInfo.householdSizes[j];
			unsigned int amount = populationInfo.ageGenderCategories[i].numPeopleInHouseholdSize[j];
			
			// Adjust this value proportionally so that our final population of people
			// will match the starting population specified in the input.
			float frac = amount / populationInfo.totalPopulationSize;
			unsigned int rounded = round(((float)amount / populationInfo.totalPopulationSize) * starting_population);
			// This loop runs once for each individual person, and so this is where we
			// generate the person agents.
			for (unsigned int k = 0; k < rounded; k++)
			{
				// Set pointer to this person.
				xmachine_memory_Person *h_person = h_person_AoS[personCounter];

				// Assign the variables for the person agent based on information from
				// the histogram.
				h_person->id = getNextID();
				h_person->gender = gender;
				h_person->householdsize = currentsize;
				h_person->busy = 0;
                                h_person->school = -1;
                                h_person->workplace = -1;

				decideAgeHivArt(h_person, dh, populationInfo, minage, maxage, gender);
				decideTransport(h_person, dh);
				decideWork(h_person, dh);
				decideBar(h_person, dh);
				decideSchool(h_person, dh);

				dh.sizesarray[h_person->id] = currentsize;

				h_person->lastinfected = -1;
				h_person->lastinfectedid = -1;
				h_person->lastinfectedtime = -1;

				
				personCounter++;
				
			}
		}
	}
}


/* This function generates household agents and decides properties about the household, such as whether the household is churchoing . It also
   allocates individual people to the household. It generates a household membership agent for each person in the household and stores the
   total number of adults in the household in an array. */
void initHouseholds(const PopulationInfo& populationInfo, DataHolder& dh) {
	unsigned int churchfreq;
	unsigned int churchgoing;

	float church_beta0 = *get_CHURCH_BETA0();
	float church_beta1 = *get_CHURCH_BETA1();

	float church_prob0 = *get_CHURCH_PROB0();
	float church_prob1 = *get_CHURCH_PROB1();
	float church_prob2 = *get_CHURCH_PROB2();
	float church_prob3 = *get_CHURCH_PROB3();
	float church_prob4 = *get_CHURCH_PROB4();
	float church_prob5 = *get_CHURCH_PROB5();
	float church_prob6 = *get_CHURCH_PROB6();

	// Compute total number of households required
	int numHouseholds = 0;
	for (unsigned int i = 0; i < populationInfo.numHouseholdSizes; i++)
		for (unsigned int j = 0; j < populationInfo.numPeopleInHouseholdSize[i] / populationInfo.householdSizes[i]; j++) { 
			numHouseholds++;
        }
	xmachine_memory_Household** h_household_AoS = h_allocate_agent_Household_array(numHouseholds);
	xmachine_memory_HouseholdMembership** h_HouseholdMembership_AoS = h_allocate_agent_HouseholdMembership_array(populationInfo.totalPopulationSize);
	unsigned int hhmembershipCounter = 0;
	unsigned int householdCounter = 0;

	// This loop runs once for each possible size of household.
	for (unsigned int i = 0; i < populationInfo.numHouseholdSizes; i++)
	{
		// This loop runs once for each individual household, as calculated from the
		// number of people living in households of each size.
		// This is presumably the total number of people who live in this household size / the number of people who live in a household of this size, to work out the number of households of this size
		for (unsigned int j = 0; j < populationInfo.numPeopleInHouseholdSize[i] / populationInfo.householdSizes[i] ; j++)
		{
			// Allocate memory for the household agent.
			xmachine_memory_Household *h_household = h_household_AoS[householdCounter];
			float churchprob = 1 / (1 + exp(-church_beta0 - church_beta1 * i));
			dh.adultcount = 0;

			// Set the household's id and size.
			h_household->id = getNextHouseholdID();
			h_household->active = 0;

			// Decide if the household is a churchgoing household, based on given
			// input probabilities.
			float random = ((float)rand() / (RAND_MAX));

			if (random < churchprob)
			{
				churchgoing = 1;
			}
			else
			{
				churchgoing = 0;
			}

			// If the household is churchgoing, decide how frequently they go based on
			// input probabilities; if not, set this variable to a dummy value.
			if (churchgoing)
			{
				random = ((float)rand() / (RAND_MAX));
				if (random < church_prob0)
				{
					churchfreq = 0;
				}
				else if (random < church_prob1)
				{
					churchfreq = 1;
				}
				else if (random < church_prob2)
				{
					churchfreq = 2;
				}
				else if (random < church_prob3)
				{
					churchfreq = 3;
				}
				else if (random < church_prob4)
				{
					churchfreq = 4;
				}
				else if (random < church_prob5)
				{
					churchfreq = 5;
				}
				else if (random < church_prob6)
				{
					churchfreq = 6;
				}
				else
				{
					churchfreq = 7;
				}
			}
			else
			{
				churchfreq = 0;
			}

			// Allocate individual people to the household until it is full, keeping
			// track of how many of them are adults.
			for (unsigned int k = 0; k < populationInfo.householdSizes[i]; k++)
			{
				xmachine_memory_HouseholdMembership *h_hhmembership = h_HouseholdMembership_AoS[hhmembershipCounter];
				h_hhmembership->household_id = h_household->id;
				h_hhmembership->person_id = dh.order[dh.count];
				h_hhmembership->churchgoing = churchgoing;
				h_hhmembership->churchfreq = churchfreq;
				h_hhmembership->household_size = i;

				if (dh.ages[dh.count] >= 15)
				{
					dh.adultcount++;
				}

				if (dh.activepeople[dh.order[dh.count]] == 1)
				{
					h_household->active = 1;
				}

				dh.count++;

				hhmembershipCounter++;
			}

			// Set the variable for how many adults belong in the household, generate
			// the agent and then free it from memory on the host.
			dh.adult[h_household->id] = dh.adultcount;

			if (h_household->active)
			{
				dh.activehouseholds[h_household->id] = 1;
			}

			householdCounter++;
		}
	}

	h_add_agents_Household_hhdefault(h_household_AoS, householdCounter);
	h_free_agent_Household_array(&h_household_AoS, householdCounter);
	h_add_agents_HouseholdMembership_hhmembershipdefault(h_HouseholdMembership_AoS, hhmembershipCounter);
	h_free_agent_HouseholdMembership_array(&h_HouseholdMembership_AoS, hhmembershipCounter);
}

void allocateTransport(const DataHolder& dh, const PopulationInfo& populationInfo) {
	float transport_dur20 = *get_TRANSPORT_DUR20();
	float transport_dur45 = *get_TRANSPORT_DUR45();

	unsigned int transport_size = *get_TRANSPORT_SIZE();

	unsigned int transportMembershipCounter = 0;

	// Allocate people to transport
	// For each day people might use transport (1-5)
	for (unsigned int i = 1; i <= 5; i++)
	{
		unsigned int currentday = 0;
		unsigned int currentpeople[h_agent_AoS_MAX];
		
		// Count how many people are travelling on this day and store them (?) in currentpeople
		for (unsigned int j = 0; j < h_agent_AoS_MAX; j++)
		{
			if (dh.days[j] == i)
			{
				currentpeople[currentday] = dh.transport[j];
				currentday++;
			}
		}

		unsigned int countdone = 0;
		unsigned int capacity = 0;

		// Until we have allocated everyone
		while (countdone < currentday)
		{
			// Create a transport agent
			xmachine_memory_Transport *h_transport = h_allocate_agent_Transport();
			capacity = 0;

			h_transport->id = getNextTransportID();

			float random = ((float)rand() / (RAND_MAX));

			// Select a journey length
			float duration;
			if (random < transport_dur20)
			{
				duration = 20;
			}
			else if (random < transport_dur45)
			{
				duration = 45;
			}
			else
			{
				duration = 60;
			}

			// While this transport agent (vehicle) has capacity and we still have people who need allocating
			while (capacity < transport_size && countdone < currentday)
			{
				// If the person we are adding is active, set the transport to activey
				if (dh.activepeople[currentpeople[countdone]] == 1)
				{
					h_transport->active = 1;
				}

				// Create a membership association between  the person and transport
				unsigned int person_id = currentpeople[countdone];
				h_person_AoS[person_id]->transport = h_transport->id;
				h_person_AoS[person_id]->transportdur = duration;

				capacity++;
				countdone++;
				transportMembershipCounter++;
			}

			h_add_agent_Transport_trdefault(h_transport);
			h_free_agent_Transport(&h_transport);
		}
	}
}


/* Allocates people to workplaces generating a workplacemembership agent for each pairing*/
void allocateWorkplaces(DataHolder& dh) {

	unsigned int workplace_size = *get_WORKPLACE_SIZE();

	// Allocate people to workplaces
	unsigned int employedpos = 0;

	shuffle(dh.workarray, dh.workarray, dh.employedcount);

	while (employedpos < dh.employedcount)
	{

		xmachine_memory_Workplace *h_workplace = h_allocate_agent_Workplace();

		h_workplace->id = getNextWorkplaceID();

		for (unsigned int j = 0; j < workplace_size; j++)
		{
			if (employedpos < dh.employedcount)
			{
				unsigned int person_id = dh.workarray[employedpos];
				h_person_AoS[person_id]->workplace = h_workplace->id;

				employedpos++;
			}
		}
	}
}

void allocateChurches(DataHolder& dh) {
	float church_p1 = *get_CHURCH_P1();
	float church_p2 = *get_CHURCH_P2();

	unsigned int church_k1 = *get_CHURCH_K1();
	unsigned int church_k2 = *get_CHURCH_K2();
	unsigned int church_k3 = *get_CHURCH_K3();

	float church_proportion = *get_CHURCH_PROPORTION();
#ifdef MEGACHURCH_TEST
       	church_proportion = 1.0f;
#endif
	// Generate an array of household ids and then shuffle it, for use when
  // generating churches and other buildings.
	unsigned int hhtotal = get_agent_Household_hhdefault_count();
	//unsigned int hhorder[xmachine_memory_Household_MAX]; // hhtotal
	unsigned int* hhorder = (unsigned int*)malloc(hhtotal * sizeof(unsigned int));

	// Allocate memory for church membership agents
	xmachine_memory_ChurchMembership **h_chumemberships =	h_allocate_agent_ChurchMembership_array(hhtotal);

	for (unsigned int i = 0; i < hhtotal; i++)
	{
		hhorder[i] = i;
	}

	shuffle(hhorder, dh.adult, hhtotal);

	for (unsigned int i = 0; i < h_household_AoS_MAX; i++)
	{
		dh.activehouseholds[i] = 0;
	}


	// Set a variable to keep track of our current position in this array.
	unsigned int hhposition = 0;
	unsigned int capacity;

	// Allocate households to churches
	while (hhposition < hhtotal)
	{

		// Allocate memory for the church agent, and set a variable to keep track of
		// how many adults have been assigned to it.
		xmachine_memory_Church *h_church = h_allocate_agent_Church();
		capacity = 0;

		h_church->id = getNextChurchID();

		// Decide what size the church is, based on given input probabilities.
		float random = ((float)rand() / (RAND_MAX));

		if (random < church_p1)
		{
			h_church->size = church_k1;
		}
		else if (random < church_p2)
		{
			h_church->size = church_k2;
		}
		else
		{
			h_church->size = church_k3;
		}

		// Decide whether the church services will be 1.5 hours or 3.5 hours, based
		// on the input probability.
		random = ((float)rand() / (RAND_MAX));

		float duration;
		if (random < church_proportion)
		{
			duration = 1.5;
		}
		else
		{
			duration = 3.5;
		}

		// Allocate households to the church until it has reached its capacity of
		// adults, as defined by the size of the church.
		dh.count = 0;
		

#ifdef MEGACHURCH_TEST
		while (hhposition < hhtotal) 
#else
		while (capacity < h_church->size && hhposition < hhtotal)
#endif
		{
			xmachine_memory_ChurchMembership *h_chumembership = h_chumemberships[hhposition];
			h_chumembership->church_id = h_church->id;
			h_chumembership->household_id = hhorder[hhposition];
			h_chumembership->churchdur = duration;

			// TODO: Commented out to ensure chuupdate function runs
			//if (dh.activehouseholds[hhorder[hhposition]] == 1)
			//{
				h_church->active = 1;
			//}

			dh.count++;
			capacity += dh.adult[hhposition];
			hhposition++;
		}

		// Generate the church agent and free it from memory on the host.
		h_add_agent_Church_chudefault(h_church);
		h_free_agent_Church(&h_church);
	}
	h_add_agents_ChurchMembership_chumembershipdefault(h_chumemberships, hhtotal);
	h_free_agent_ChurchMembership_array(&h_chumemberships, hhtotal);
	free(hhorder);
}

void allocateBars(DataHolder& dh) {
	unsigned int bar_size = *get_BAR_SIZE();

	// Allocate people to bars? Not required, people go to random bars
	for (unsigned int i = 0; i < dh.barcount / bar_size; i++)
	{
		xmachine_memory_Bar *h_bar = h_allocate_agent_Bar();

		h_bar->id = getNextBarID();

		h_add_agent_Bar_bdefault(h_bar);

		h_free_agent_Bar(&h_bar);
	}

}

void allocateSchools(DataHolder& dh) {
	// Allocate children to schools
	unsigned int school_size = *get_SCHOOL_SIZE();
	shuffle(dh.schoolarray, dh.schoolarray, dh.childcount);
	unsigned int childpos = 0;

	unsigned int numSchools = dh.childcount / school_size;
	xmachine_memory_School **h_schools = h_allocate_agent_School_array(numSchools);

	for (unsigned int i = 0; i < dh.childcount / school_size; i++)
	{
		xmachine_memory_School *h_school = h_schools[i];

		h_school->id = getNextSchoolID();

		for (unsigned int j = 0; j < school_size; j++)
		{
			if (childpos <= dh.childcount)
			{
				unsigned int person_id = dh.schoolarray[childpos];
				// Think person IDs start from 1 - check this
				h_person_AoS[person_id]->school = h_school->id;
				childpos++;		
			}
		}
	}
	h_add_agents_School_schdefault(h_schools, numSchools);
	h_free_agent_School_array(&h_schools, numSchools);
}

void allocateTB(DataHolder& dh) {
	float tb_prevalence = *get_TB_PREVALENCE();

	for (unsigned int i = 0; i < dh.total; i++)
	{
		dh.tbarray[i] = i + 1;
	}

	float weightsum = 0;

	for (unsigned int i = 0; i < dh.total; i++)
	{
		weightsum += dh.weights[i];
	}

	unsigned int tbnumber = ceil(dh.total * tb_prevalence);

	dh.activepeople = (unsigned int*)malloc(dh.total * sizeof(unsigned int));

	for (unsigned int i = 0; i < dh.total; i++)
	{
		dh.activepeople[i] = 0;
	}

	for (unsigned int i = 0; i < tbnumber; i++)
	{

		float randomweight = weightsum * ((float)rand() / (RAND_MAX));

		for (unsigned int j = 0; j < dh.total; j++)
		{
			if (randomweight < dh.weights[j])
			{

				xmachine_memory_TBAssignment *h_tbassignment =
					h_allocate_agent_TBAssignment();

				h_tbassignment->id = dh.tbarray[j];
				dh.activepeople[dh.tbarray[j]] = 1;
				weightsum -= dh.weights[j];
				dh.weights[j] = 0.0;

				h_add_agent_TBAssignment_tbdefault(h_tbassignment);

				h_free_agent_TBAssignment(&h_tbassignment);
				break;
			}

			randomweight -= dh.weights[j];
		}
	}
}

void allocateClinics() {

	xmachine_memory_Clinic *h_clinic = h_allocate_agent_Clinic();
	h_clinic->id = 1;
	h_add_agent_Clinic_cldefault(h_clinic);
	h_free_agent_Clinic(&h_clinic);
}

void setConstants() {
	unsigned int householdcount = get_agent_Household_hhdefault_count();
	set_HOUSEHOLDS(&householdcount);

	unsigned int bars = get_agent_Bar_bdefault_count();
	set_BARS(&bars);

	float time_step = *get_TIME_STEP();

	float household_a = *get_HOUSEHOLD_A();
	float church_a = *get_CHURCH_A();
	float transport_a = *get_TRANSPORT_A();
	float clinic_a = *get_CLINIC_A();
	float workplace_a = *get_WORKPLACE_A();
	float bar_a = *get_BAR_A();
	float school_a = *get_SCHOOL_A();

	float household_exp = exp(-household_a * (time_step / 12));
	float church_exp = exp(-church_a * (time_step / 12));
	float transport_exp = exp(-transport_a * (time_step / 12));
	float clinic_exp = exp(-clinic_a * (time_step / 12));
	float workplace_exp = exp(-workplace_a * (time_step / 12));
	float bar_exp = exp(-bar_a * (time_step / 12));
	float school_exp = exp(-school_a * (time_step / 12));
	// TODO: Check if this is correct - unlikely as it gives negative probability
	float prob = 1.0f;// 1 - exp(6.0 / 365);

	set_HOUSEHOLD_EXP(&household_exp);
	set_CHURCH_EXP(&church_exp);
	set_TRANSPORT_EXP(&transport_exp);
	set_CLINIC_EXP(&clinic_exp);
	set_WORKPLACE_EXP(&workplace_exp);
	set_BAR_EXP(&bar_exp);
	set_SCHOOL_EXP(&school_exp);
	set_PROB(&prob);

	float theta = *get_DEFAULT_Q() / *get_DEFAULT_K();

	set_THETA(&theta);

	float e = exp(1.0);
	set_E(&e);


	// Set location edge offsets for network based messaging
	// Get number of each type of location agent
	unsigned int h_LOCATION_EDGE_OFFSETS[10];
	unsigned int numHouseholds = get_agent_Household_hhdefault_count();
	unsigned int numChurches = get_agent_Church_chudefault_count();
	unsigned int numTransport = get_agent_Transport_trdefault_count();
	unsigned int numClinics = get_agent_Clinic_cldefault_count();
	unsigned int numWorkplaces = get_agent_Workplace_wpdefault_count();
	unsigned int numBars = get_agent_Bar_bdefault_count();
	unsigned int numSchools = get_agent_School_schdefault_count();
	h_LOCATION_EDGE_OFFSETS[0] = 0;
	h_LOCATION_EDGE_OFFSETS[1] = numHouseholds;
	h_LOCATION_EDGE_OFFSETS[2] = h_LOCATION_EDGE_OFFSETS[1] + numChurches;
	h_LOCATION_EDGE_OFFSETS[3] = h_LOCATION_EDGE_OFFSETS[2] + numTransport;
	h_LOCATION_EDGE_OFFSETS[4] = h_LOCATION_EDGE_OFFSETS[3] + numClinics;
	h_LOCATION_EDGE_OFFSETS[5] = h_LOCATION_EDGE_OFFSETS[4] + numWorkplaces;
	h_LOCATION_EDGE_OFFSETS[6] = h_LOCATION_EDGE_OFFSETS[5] + numBars;
	h_LOCATION_EDGE_OFFSETS[7] = h_LOCATION_EDGE_OFFSETS[6] + numSchools;
	set_LOCATION_EDGE_OFFSETS(h_LOCATION_EDGE_OFFSETS);
}

// The function called at the beginning of the program on the CPU, to initialise
// all agents and their corresponding variables.
__FLAME_GPU_INIT_FUNC__ void initialiseHost()
{
  unsigned int seed = *get_SEED();
  srand(seed);

  PopulationInfo populationInfo = readInputFile();
  DataHolder dh;

  initAgentTypes();

  // Set a counter for our current position in the array of person ids, to
  // keep track as we generate households.
  dh.count = 0;
  dh.total = populationInfo.totalPopulationSize;
  dh.order = (unsigned int*)malloc(dh.total * sizeof(unsigned int));


  // Populate the array of person ids with ids up to the total number of people,
  // and then shuffle it so households are assigned randomly.
  //for (unsigned int i = 1; i <= total; i++)
  for (unsigned int i = 0; i < dh.total; i++)
  {
    dh.order[i] = i;
  }

  //shuffle(dh.order, dh.sizesarray, dh.total);
  //quickSort(dh.sizesarray, dh.order, 0, dh.total);
  initPeople(populationInfo, dh);

  shuffle(dh.transport, dh.days, dh.total);


  allocateTB(dh);
  initHouseholds(populationInfo, dh);
  allocateChurches(dh);
#ifndef MEGACHURCH_TEST
  allocateTransport(dh, populationInfo);
  allocateClinics();
  allocateWorkplaces(dh);
  allocateBars(dh);
  allocateSchools(dh);
#endif

  setConstants();

  // deallocating the memory - people are not added until the end as allocate functions are still setting memberships
  h_add_agents_Person_default(h_person_AoS, personCounter);
  h_free_agent_Person_array(&h_person_AoS, personCounter);

  free(dh.order);
  free(dh.activepeople);
}

// Function that prints out the number of agents generated after initialisation.
__FLAME_GPU_INIT_FUNC__ void generateAgentsInit()
{
  printf("Population after init function: %u\n",
         get_agent_Person_default_count());
}

// Function for generating output data in csv files, which runs after every
// iteration and saves data whenever specified.
__FLAME_GPU_EXIT_FUNC__ void customOutputFunction()
{

  // Array to track when infections took place
	std::vector<int> infectionFreq;
	infectionFreq.resize(8000);

  // Assign a variable for the directory where our files will be output, and
  // check which iteration we are currently on.
  const char *directory = getOutputDir();

  int output_id = *get_OUTPUT_ID();

  // If there is new information about the person agents to output, this code
  // creates a csv file and outputs data about people and their variables to
  // that file.
  std::string outputFilename =
      std::string(std::string(directory) + "person-output-" + std::to_string(output_id) + ".csv");

  FILE *fp = fopen(outputFilename.c_str(), "w");

  if (fp != nullptr)
  {
    fprintf(stdout, "Outputting some Person data to %s\n",
            outputFilename.c_str());

    fprintf(fp, "ID, gender, age, household_size, household, church, "
                "workplace, school, hiv, art, "
                "active_tb, "
                "time_home, time_visiting, time_church, "
                "time_transport, time_clinic, time_workplace, time_bar, "
                "time_school, time_outside, "
                "infections, "
                "last_infected_type, "
                "last_infected_id, last_infected_time\n");

    for (int index = 0; index < get_agent_Person_s2_count(); index++)
    {

      fprintf(fp,
              "%u, %u, %u, %u, %u, %i, %i, %i, %u, %u, %u, %u, %u, %u, %u, %u, "
              "%u, %u, %u, %u, %u, %i, %i, %i\n",
              get_Person_s2_variable_id(index),
              get_Person_s2_variable_gender(index),
              get_Person_s2_variable_age(index),
              get_Person_s2_variable_householdsize(index),
              get_Person_s2_variable_household(index),
              get_Person_s2_variable_church(index),
              get_Person_s2_variable_workplace(index),
              get_Person_s2_variable_school(index),
              get_Person_s2_variable_hiv(index),
              get_Person_s2_variable_art(index),
              get_Person_s2_variable_activetb(index),
              get_Person_s2_variable_householdtime(index),
              get_Person_s2_variable_timevisiting(index),
              get_Person_s2_variable_churchtime(index),
              get_Person_s2_variable_transporttime(index),
              get_Person_s2_variable_clinictime(index),
              get_Person_s2_variable_workplacetime(index),
              get_Person_s2_variable_bartime(index),
              get_Person_s2_variable_schooltime(index),
              get_Person_s2_variable_outsidetime(index),
              get_Person_s2_variable_infections(index),
              get_Person_s2_variable_lastinfected(index),
              get_Person_s2_variable_lastinfectedid(index),
              get_Person_s2_variable_lastinfectedtime(index));

	  int infectedtime = get_Person_s2_variable_lastinfectedtime(index);
	  if (infectedtime >= 0) {
		  infectionFreq[infectedtime / 5] += 1;
	  }
    }
	
	

    fflush(fp);
  }
  else
  {
    fprintf(
        stderr,
        "Error: file %s could not be created for customOutputStepFunction\n",
        outputFilename.c_str());
  }

  if (fp != nullptr && fp != stdout && fp != stderr)
  {
    fclose(fp);
    fp = nullptr;
  }

  // Compute cumulative frequency for infections
  for (int i = 1; i < infectionFreq.size(); i++) {
	  infectionFreq[i] = infectionFreq[i - 1] + infectionFreq[i];
  }

  // Save c.f. to file
  std::ofstream cfFile("iterations/infection_cf.csv");
  if (cfFile.is_open()) {
	  for (int i = 0; i < infectionFreq.size(); i++) {
		  cfFile << infectionFreq[i] << std::endl;
	  }
  }


}

// At the end of the run, free all of the agents from memory and output the
// final population of people.
__FLAME_GPU_EXIT_FUNC__ void exitFunction()
{

  h_free_agent_Person_array(&h_agent_AoS, h_agent_AoS_MAX);
  h_free_agent_Household_array(&h_household_AoS, h_household_AoS_MAX);
  h_free_agent_Church_array(&h_church_AoS, h_church_AoS_MAX);
  h_free_agent_Transport_array(&h_transport_AoS, h_transport_AoS_MAX);
  h_free_agent_Workplace_array(&h_workplace_AoS, h_workplace_AoS_MAX);
  h_free_agent_Bar_array(&h_bar_AoS, h_bar_AoS_MAX);
  h_free_agent_TBAssignment_array(&h_tbassignment_AoS, h_tbassignment_AoS_MAX);
  h_free_agent_ChurchMembership_array(&h_chumembership_AoS,
                                      h_chumembership_AoS_MAX);


  unsigned int population =
      get_agent_Person_s2_count() + get_agent_Household_hhdefault_count() +
      get_agent_Church_chudefault_count() +
      get_agent_Transport_trdefault_count() +
      get_agent_Workplace_wpdefault_count() + get_agent_Bar_bdefault_count() +
      get_agent_School_schdefault_count();

  printf("Population for exit function: %u\n", population);
}

// The update functions for each agent type, which are involved in deciding
// where a person is at a given time.

/* This function updates a person agent by deciding what they are doing/where they are this timestep*/
__FLAME_GPU_FUNC__ int update(xmachine_memory_Person *person,
                              xmachine_message_location_list *location_messages,
                              RNG_rand48 *rand48)
{

  unsigned int day = dayofweek(person->step);
  unsigned int monthday = dayofmonth(person->step);
  struct Time t = timeofday(person->step);
  unsigned int hour = t.hour;
  unsigned int minute = t.minute;

  // If the person isn't busy, decide what they are going to do
  if (person->busy == 0)
  {
	// If 8PM & person is a church-goer
    if (hour == 20 && minute == 0 && person->church != -1)
    {
      if (person->churchfreq == 0)
      {
        float random = rnd<CONTINUOUS>(rand48);

        if (random < PROB) // Person goes to church
        {
          person->startstep = person->step;
          person->busy = 1;
          person->location = CHURCH;
          person->locationid = person->church;
        }
        else // Person stays at home
        {
          person->location = HOUSEHOLD;
          person->locationid = person->household;
        }
      }
      else if (person->churchfreq > day)
      {
        person->startstep = person->step;
        person->busy = 1;
        person->location = 1;
        person->locationid = person->church;
      }
    }
#ifndef MEGACHURCH_TEST
	// If it's a transport day and the person uses transport
    else if (person->transportdur != 0 &&
             (day == person->transportday1 || day == person->transportday2))
    {
      if ((hour == 7 && minute == 0) || (hour == 18 && minute == 0)) // Person takes journey if it is the right time of day
      {
        person->startstep = person->step;
        person->busy = 1;
        person->location = TRANSPORT;
        person->locationid = person->transport;
      }
      else // Otherwise person stays at home
      {
        person->location = HOUSEHOLD;
        person->locationid = person->household;
      }
    }
	// Should person go to clinic
    else if (person->art == 1 && monthday == person->artday && hour == 9 &&
             minute == 0)
    {
      person->startstep = person->step;
      person->busy = 1;
      person->location = CLINIC;
      person->locationid = 1;
    }
	// Go to work
    else if (person->workplace != -1 && day != 0 && day != 6 && hour == 9 &&
             minute == 0)
    {
      person->startstep = person->step;
      person->busy = 1;
      person->location = WORKPLACE;
      person->locationid = person->workplace;
    }
	// Visit another house
    else if (hour == 0 && minute == 0)
    {
      float prob = 0.23;
      float random = rnd<CONTINUOUS>(rand48);

      if (random < prob)
      {
        float random = rnd<CONTINUOUS>(rand48);
        unsigned int randomhouse = ceil(random * HOUSEHOLDS);

        person->startstep = person->step;
        person->busy = 1;
        person->location = HOUSEHOLD;
        person->locationid = randomhouse;
      }
    }
	// Visit a bar
    else if (person->bargoing != 0 && hour == 1 && minute == 0)
    {
      if (person->bargoing == 1 && person->barday > day)
      {
        person->startstep = person->step;
        person->busy = 1;
        person->location = BAR;

		// Select a random bar to go to
        float random = rnd<CONTINUOUS>(rand48);
        unsigned int randombar = ceil(random * BARS);

        person->locationid = randombar;
      }
      else if (person->bargoing == 2 && person->barday == monthday)
      {
        person->startstep = person->step;
        person->busy = 1;
        person->location = BAR;

        float random = rnd<CONTINUOUS>(rand48);
        unsigned int randombar = ceil(random * BARS);

        person->locationid = randombar;
      }
      else // If not going to the bar today, stay at home
      {
        person->location = HOUSEHOLD;
        person->locationid = person->household;
      }
    }
	// See if person should go to school
    else if (person->school != -1 && hour == 9 && minute == 0)
    {
      person->startstep = person->step;
      person->busy = 1;
      person->location = SCHOOL;
      person->locationid = person->school;
    }
	// If between 8pm and 6am, person is in the house
    else if (hour >= 20 || hour <= 6)
    {
      person->location = HOUSEHOLD;
      person->locationid = person->household;
    }
    else // Otherwise, person is outside
    {
      person->location = OUTSIDE;
      person->locationid = 0;
    }
#endif
  }
  else // Person is busy already
  {
	// If person is at church and the service has finished, go home
    if (person->location == CHURCH &&
        (float)(person->step - person->startstep) >= person->churchdur * 12)
    {
      person->busy = 0;
      person->location = HOUSEHOLD;
      person->locationid = person->household;
    }
	// If transport journey complete, person now goes outside
    else if (person->location == TRANSPORT &&
             (float)(person->step - person->startstep) >=
                 person->transportdur / 5)
    {
      person->busy = 0;
      person->location = OUTSIDE;
      person->locationid = 0;
    }
	// If person is finished at the clinic, go outside
    else if (person->location == CLINIC &&
             (float)(person->step - person->startstep) >=
                 (CLINIC_DUR * 12 * TIME_STEP))
    {
      person->busy = 0;
      person->location = OUTSIDE;
      person->locationid = 0;
    }
	// If the person is finished at work, go outside
    else if (person->location == WORKPLACE &&
             (float)(person->step - person->startstep) >=
                 WORKPLACE_DUR * 12 * TIME_STEP)
    {
      person->busy = 0;
      person->location = OUTSIDE;
      person->locationid = 0;
    }
	// If the person has finished at the bar, go home
    else if (person->location == BAR &&
             (float)(person->step - person->startstep) >=
                 ((BAR_DUR / 5) * TIME_STEP))
    {
      person->busy = 0;
      person->location = HOUSEHOLD;
      person->locationid = person->household;
    }
	// If the person has finished at school, go outside
    else if (person->location == SCHOOL &&
             (float)(person->step - person->startstep) >=
                 (SCHOOL_DUR * 12 * TIME_STEP))
    {
      person->busy = 0;
      person->location = OUTSIDE;
      person->locationid = 0;
    }
	// If the person has finished visiting another household, go home
    else if (person->location == HOUSEHOLD &&
             (float)(person->step - person->startstep) >=
                 ((VISITING_DUR / 12) * TIME_STEP))
    {
      person->busy = 0;
      person->locationid = person->household;
    }
  }


  // Track time spent at the person's location for this timestep
  if (person->location == HOUSEHOLD && person->busy == 1)
  {
    person->timevisiting += 5 * TIME_STEP;
  }
  else if (person->location == HOUSEHOLD)
  {
    person->householdtime += 5 * TIME_STEP;
  }
  else if (person->location == CHURCH)
  {
    person->churchtime += 5 * TIME_STEP;
  }
  else if (person->location == TRANSPORT)
  {
    person->transporttime += 5 * TIME_STEP;
  }
  else if (person->location == CLINIC)
  {
    person->clinictime += 5 * TIME_STEP;
  }
  else if (person->location == WORKPLACE)
  {
    person->workplacetime += 5 * TIME_STEP;
  }
  else if (person->location == BAR)
  {
    person->bartime += 5 * TIME_STEP;
  }
  else if (person->location == SCHOOL)
  {
    person->schooltime += 5 * TIME_STEP;
  }
  else if (person->location == OUTSIDE)
  {
    person->outsidetime += 5 * TIME_STEP;
  }

#ifdef MEGACHURCH_TEST
  person->location = CHURCH;
  person->locationid = 0;//person->church;
#endif

  // If the person is an active carrier of TB, add a message stating their location
  if (person->activetb == 1)
  {
	  
    add_location_message(location_messages, person->id, person->location,
                         person->locationid, person->p, person->q, locationToEdgeID(person->location, person->locationid));
  }

  person->step += TIME_STEP;

  return 0;
}

__FLAME_GPU_FUNC__ int 
updatelambda(xmachine_memory_Person *person,
	xmachine_message_infection_list *infection_messages,
	xmachine_message_infection_bounds* message_bounds)
{
	// Each location only outputs a single message so we know there will only be one message
	xmachine_message_infection *infection_message =
		get_first_infection_message(infection_messages, message_bounds, locationToEdgeID(person->location, person->locationid));
	if (infection_message) {
		person->lambda = infection_message->lambda;
	}
	return 0;
}


/* Computes whether a person has become infected based on their lambda value & 'p' value. */
__FLAME_GPU_FUNC__ int infect(xmachine_memory_Person *person,
                              RNG_rand48 *rand48)
{

  float prob = 1 - device_exp(-person->p * person->lambda * (TIME_STEP / 12));
  float random = rnd<CONTINUOUS>(rand48);
  //printf("Infection prob %f, p: %f, lambda: %f \n", prob, person->p, person->lambda);
  if (random < prob)
  {
    person->infections++;
    person->lastinfected = person->location;
    person->lastinfectedid = person->locationid;
    person->lastinfectedtime = person->step * 5 * TIME_STEP;
  }

  return 0;
}
/* Each locationtype update function computes a 'qsum' from location messages which is then used to compute the lambda value for the location. 
   Location messages come from each agent, and the q value is also that of the person. Hence each location here is summing the q values of the
   person agents who are at that location to compute qsum. Each locationtype agent then outputs an infection message. 
   
   location_message->location is the type of place, e.g. bar, church, household. */
__FLAME_GPU_FUNC__ int
hhupdate(xmachine_memory_Household *household,
         xmachine_message_location_list *location_messages,
		 xmachine_message_location_bounds* message_bounds,
         xmachine_message_infection_list *infection_messages)
{

  float qsum = 0.0;

  xmachine_message_location *location_message =
      get_first_location_message(location_messages, message_bounds, locationToEdgeID(HOUSEHOLD, household->id));

  while (location_message)
  {
    qsum += location_message->q;
    //printf("%u", qsum); // not required
    location_message =
        get_next_location_message(location_message, location_messages, message_bounds);
  }

  household->lambda =
      (household->lambda * HOUSEHOLD_EXP) +
      ((qsum / (HOUSEHOLD_V * HOUSEHOLD_A)) * (1 - HOUSEHOLD_EXP));

  add_infection_message(infection_messages, household->id,
                                  household->lambda, locationToEdgeID(HOUSEHOLD, household->id));

  return 0;
}

__FLAME_GPU_FUNC__ int
chuupdate(xmachine_memory_Church *church,
          xmachine_message_location_list *location_messages,
		  xmachine_message_location_bounds* message_bounds,
          xmachine_message_infection_list *infection_messages)
{
  
  float qsum = 0;
  //printf("CHURCH, church->id, locationToEdgeID: %u, %u, %u \n", CHURCH, church->id, locationToEdgeID(CHURCH, church->id));
  xmachine_message_location *location_message =
      get_first_location_message(location_messages, message_bounds, locationToEdgeID(CHURCH, church->id));

   while (location_message)
  {
	   //printf("q: %f \n", location_message->q);
    qsum += location_message->q;
    location_message =
        get_next_location_message(location_message, location_messages, message_bounds);
  }

  //printf("Church qsum: %f", qsum);

  // Lambda computed as: 
  // Previous lambda * decay factor + scaled qsum * (1-decay factor) 
  // So really dependent on qsum, from location messages
  church->lambda = (church->lambda * CHURCH_EXP) +
                   ((qsum / (CHURCH_V_MULTIPLIER * church->size * CHURCH_A)) *
                    (1 - CHURCH_EXP));
  //printf("Church lambda: %f", church->lambda);
  add_infection_message(infection_messages, church->id, church->lambda, locationToEdgeID(CHURCH, church->id));

  return 0;
}

__FLAME_GPU_FUNC__ int
trupdate(xmachine_memory_Transport *transport,
         xmachine_message_location_list *location_messages,
		 xmachine_message_location_bounds* message_bounds,
         xmachine_message_infection_list *infection_messages)
{

  float qsum = 0;

  xmachine_message_location *location_message =
      get_first_location_message(location_messages, message_bounds, locationToEdgeID(TRANSPORT, transport->id));

  while (location_message)
  {
    qsum += location_message->q;
    location_message =
        get_next_location_message(location_message, location_messages, message_bounds);
  }

  transport->lambda =
      (transport->lambda * TRANSPORT_EXP) +
      ((qsum / (TRANSPORT_V * TRANSPORT_A)) * (1 - TRANSPORT_EXP));

  add_infection_message(infection_messages, transport->id,
                                  transport->lambda, locationToEdgeID(TRANSPORT, transport->id));

  return 0;
}

__FLAME_GPU_FUNC__ int
clupdate(xmachine_memory_Clinic *clinic,
         xmachine_message_location_list *location_messages,
		 xmachine_message_location_bounds* message_bounds,
         xmachine_message_infection_list *infection_messages)
{

  float qsum = 0;

  xmachine_message_location *location_message =
	  get_first_location_message(location_messages, message_bounds, locationToEdgeID(CLINIC, clinic ->id));

  while (location_message)
  {
    qsum += location_message->q;
    location_message =
        get_next_location_message(location_message, location_messages, message_bounds);
  }

  clinic->lambda = (clinic->lambda * CLINIC_EXP) +
                   ((qsum / (CLINIC_V * CLINIC_A)) * (1 - CLINIC_EXP));

  add_infection_message(infection_messages, clinic->id, clinic->lambda, locationToEdgeID(CLINIC, clinic->id));

  return 0;
}

__FLAME_GPU_FUNC__ int
wpupdate(xmachine_memory_Workplace *workplace,
         xmachine_message_location_list *location_messages,
	     xmachine_message_location_bounds* message_bounds,
         xmachine_message_infection_list *infection_messages)
{

  float qsum = 0;

  xmachine_message_location *location_message =
      get_first_location_message(location_messages, message_bounds, locationToEdgeID(WORKPLACE, workplace->id));

  while (location_message)
  {
    qsum += location_message->q;
    location_message =
        get_next_location_message(location_message, location_messages, message_bounds);
  }

  workplace->lambda =
      (workplace->lambda * WORKPLACE_EXP) +
      ((qsum / (WORKPLACE_V * WORKPLACE_A)) * (1 - WORKPLACE_EXP));

  add_infection_message(infection_messages, workplace->id,
                                  workplace->lambda, locationToEdgeID(WORKPLACE, workplace->id));

  return 0;
}

__FLAME_GPU_FUNC__ int
bupdate(xmachine_memory_Bar *bar,
        xmachine_message_location_list *location_messages,
		xmachine_message_location_bounds* message_bounds,
        xmachine_message_infection_list *infection_messages)
{

  float qsum = 0;

  xmachine_message_location *location_message =
      get_first_location_message(location_messages, message_bounds, locationToEdgeID(BAR, bar->id));

  while (location_message)
  {
    qsum += location_message->q;
    location_message =
        get_next_location_message(location_message, location_messages, message_bounds);
  }

  bar->lambda =
      (bar->lambda * BAR_EXP) + ((qsum / (BAR_V * BAR_A)) * (1 - BAR_EXP));

  add_infection_message(infection_messages, bar->id, bar->lambda, locationToEdgeID(BAR, bar->id));

  return 0;
}

__FLAME_GPU_FUNC__ int
schupdate(xmachine_memory_School *school,
          xmachine_message_location_list *location_messages,
		  xmachine_message_location_bounds* message_bounds,
          xmachine_message_infection_list *infection_messages)
{

  float qsum = 0;

  xmachine_message_location *location_message =
      get_first_location_message(location_messages, message_bounds, locationToEdgeID(SCHOOL, school->id));

  while (location_message)
  {
    qsum += location_message->q;
    location_message =
        get_next_location_message(location_message, location_messages, message_bounds);
  }

  school->lambda = (school->lambda * SCHOOL_EXP) +
                   ((qsum / (SCHOOL_V * SCHOOL_A)) * (1 - SCHOOL_EXP));

  add_infection_message(infection_messages, school->id, school->lambda, locationToEdgeID(SCHOOL, school->id));

  return 0;
}

/* Each of these init functions simply outputs an assignment message. There is a single agent for every relation between a place and person which outputs this each step.
   These return 1 killing the agent at the end of the step, so the messages aren't actually repeated. */
__FLAME_GPU_FUNC__ int
tbinit(xmachine_memory_TBAssignment *tbassignment,
       xmachine_message_tb_assignment_list *tb_assignment_messages)
{
  add_tb_assignment_message(tb_assignment_messages, tbassignment->id);
  return 1;
}


__FLAME_GPU_FUNC__ int
chuinit(xmachine_memory_ChurchMembership *chumembership,
        xmachine_message_church_membership_list *church_membership_messages)
{
  add_church_membership_message(
      church_membership_messages, chumembership->church_id,
      chumembership->household_id, chumembership->churchdur);
  return 1;
}

__FLAME_GPU_FUNC__ int hhinit(
    xmachine_memory_HouseholdMembership *hhmembership,
    xmachine_message_church_membership_list *church_membership_messages,
    xmachine_message_household_membership_list *household_membership_messages)
{

  int churchid = -1;
  float churchdur = 0;
  xmachine_message_church_membership *church_membership_message =
      get_first_church_membership_message(church_membership_messages);
  unsigned int householdid = hhmembership->household_id;

  while (church_membership_message)
  {
    if (church_membership_message->household_id == householdid &&
        hhmembership->churchgoing)
    {
      churchid = (int)church_membership_message->church_id;
      churchdur = church_membership_message->churchdur;
    }
    church_membership_message = get_next_church_membership_message(
        church_membership_message, church_membership_messages);
  }
  add_household_membership_message(
      household_membership_messages, hhmembership->household_id,
      hhmembership->person_id, hhmembership->household_size, churchid,
      hhmembership->churchfreq, churchdur);
  return 1;
}

__FLAME_GPU_FUNC__ int
persontbinit(xmachine_memory_Person *person,
             xmachine_message_tb_assignment_list *tb_assignment_messages,
             RNG_rand48 *rand48)
{
  unsigned int personid = person->id;
  float usum = 0.0f;

  if (person->gender == 1)
  {
    person->p = DEFAULT_M_P;
    //printf("Male init\n");
  }
  else
  {
    person->p = DEFAULT_F_P;
    //printf("Female init\n");
  }

  for (unsigned int i = 0; i < DEFAULT_K; i++)
  {
    float random = rnd<CONTINUOUS>(rand48);

    float u = log(random);

    usum += u;
  }

  //printf("usum: %f\n", usum);
  float u = rnd<CONTINUOUS>(rand48);
  float v = rnd<CONTINUOUS>(rand48);
  float w = rnd<CONTINUOUS>(rand48);

  // float dev = cadgamma(DELTA, u, v, w);
  float dev = cadgamma(DELTA, u, v, w, 1000);
  usum += dev;

  person->q = -usum * THETA;

  xmachine_message_tb_assignment *tb_assignment_message =
      get_first_tb_assignment_message(tb_assignment_messages);

  while (tb_assignment_message)
  {
    if (tb_assignment_message->id == personid)
    {
      person->activetb = 1;
    }
    tb_assignment_message = get_next_tb_assignment_message(
        tb_assignment_message, tb_assignment_messages);
  }

  return 0;
}

/* These init functions scan the entire list of membership messages to set the person agent's membership values - these don't change through a sim so should bet set at initialisation instead. */

__FLAME_GPU_FUNC__ int personhhinit(
    xmachine_memory_Person *person,
    xmachine_message_household_membership_list *household_membership_messages)
{
  xmachine_message_household_membership *household_membership_message =
      get_first_household_membership_message(household_membership_messages);
  unsigned int personid = person->id;
  while (household_membership_message)
  {
    if (household_membership_message->person_id == personid)
    {
      person->household = household_membership_message->household_id;
      // person->householdsize = household_membership_message->household_size;
      person->church = household_membership_message->church_id;
      person->churchfreq = household_membership_message->churchfreq;
      person->churchdur = household_membership_message->churchdur;
    }
    household_membership_message = get_next_household_membership_message(
        household_membership_message, household_membership_messages);
  }
  return 0;
}

#endif
