package coursework;

import java.util.ArrayList;

import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

/**
 * Implements a basic Evolutionary Algorithm to train a Neural Network
 * 
 * You Can Use This Class to implement your EA or implement your own class that
 * extends {@link NeuralNetwork}
 * 
 */


public class ExampleEvolutionaryAlgorithm extends NeuralNetwork
{

	public enum Selection{Random, Stochastic ,Tournament}

	public enum Crossover{OnePoint,NPoint,Clone,Uniform,}

	public enum Replacement{Random,Tournament,Worst}
	
	public static Selection Selectyion = Selection.Tournament;
	public static int totalFitness = 0; 
	public static Crossover CrossOver = Crossover.Uniform;
	public static Replacement Replace = Replacement.Random;
	
	int firstCheck = -1;

	/**
	 * The Main Evolutionary Loop
	 */
	@Override
	public void run()
	{
		

		// Initialise a population of Individuals with random weights
		population = initialise();

		// Record a copy of the best Individual in the population
		best = getBest();
		System.out.println("Best From Initialisation " + best);

		/**
		 * main EA processing loop
		 */
		while (evaluations < Parameters.maxEvaluations)
		{

			/**
			 * this is a skeleton EA - you need to add the methods. You can also change the
			 * EA if you want You must set the best Individual at the end of a run
			 * 
			 */

			// Select 2 Individuals from the current population.
			Individual parent1 = select();
			Individual parent2 = select();

			// Generate a child by crossover.
			ArrayList<Individual> children = reproduce(parent1, parent2);

			// mutate the offspring
			mutate(children);

			// Evaluate the children
			evaluateIndividuals(children);

			// Replace children in population
			replace(children);

			// check to see if the best has improved
			best = getBest();

			// Implemented in NN class.
			outputStats();

			// Increment number of completed generations
		}

		// save the trained network to disk
		saveNeuralNetwork();
	}

	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 * 
	 */
	private void evaluateIndividuals(ArrayList<Individual> individuals)
	{
		for (Individual individual : individuals)
		{
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}

	/**
	 * Returns a copy of the best individual in the population
	 * 
	 */
	private Individual getBest()
	{
		best = null;
		;
		for (Individual individual : population)
		{
			if (best == null)
			{
				best = individual.copy();
			}
			else if (individual.fitness < best.fitness)
			{
				best = individual.copy();
			}
		}
		return best;
	}

	/**
	 * Generates a randomly initialised population
	 * 
	 */
	private ArrayList<Individual> initialise()
	{
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i)
		{
			// chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}
		evaluateIndividuals(population);
		return population;
	}
	//tournement function
	private int Tournament(int first, int second)

	{
		if (population.get(first).compareTo(population.get(second)) == -1)
		{
			return first;
		}
		else
		{
			return second;
		}
	}
	//failed attempt at stochastic
	//private Individual StochasticSelect() 
	//{
		//ArrayList<Integer> indexArray = new ArrayList<Integer>();
		//for (int i = 0; i < Parameters.popSize; i++)
	//	{
	//		totalFitness += population.get(i).fitness;
	//	}

		//Individual parent = new Induvidual();

	//	return parent.copy();
	//}
	//allows a tournement ot occur during select phase
	private Individual TournamentSelect()
	{
		ArrayList<Integer> indexArray = new ArrayList<Integer>();
		for (int i = 0; i < Parameters.popSize; i++)
		{
			indexArray.add(i);
		}
		if (firstCheck != -1)
		{
			indexArray.remove(firstCheck);
		}
		int firstFighter = indexArray.get((int) (Math.random() * indexArray.size()));
		int best = 0;
		indexArray.remove((int) (Math.random() * indexArray.size()));

		for (int i = 0; i < Parameters.tSize; i++)
		{

			int SecondFighter = indexArray.get((int) (Math.random() * indexArray.size()));
			indexArray.remove((int) (Math.random() * indexArray.size()));
			best = Tournament(firstFighter, SecondFighter);

		}
		Individual parent = population.get(best);
		firstCheck = best;

		return parent.copy();
	}
	
	private Individual select()
	{

		Individual Parent = population.get(Parameters.random.nextInt(Parameters.popSize));
		switch (Selectyion)
		{
			case Random:
				break;
			case Tournament:
				Parent = TournamentSelect();
				break;
			case Stochastic:
				//Parent = StochasticSelect();
				break;
			default:
				break;
		}
		return Parent.copy();
	}
//performs the uniform crossover 
	private ArrayList<Individual> UniformCrossOver(Individual FirstParent, Individual SecondParent)
	{
		ArrayList<Individual> Children = new ArrayList<>();
		Individual Child = new Individual();
		double[] Chromosome = new double[Parameters.getNumGenes()];
		for (int i = 0; i < Parameters.getNumGenes(); i++)
		{
			if (Math.random() > Parameters.uniformProb)
			{
				Chromosome[i] = SecondParent.chromosome[i];
			}
			else
			{
				Chromosome[i] = FirstParent.chromosome[i];
			}
		}
		Child.chromosome = Chromosome;
		Children.add(Child);
		return Children;
	}

	private ArrayList<Individual> OnePointCrossOver(Individual FirstParent, Individual SecondParent)
	{
		ArrayList<Individual> children = new ArrayList<>();
		Individual child = new Individual();
		boolean parent = false;
		double[] chromosome = new double[Parameters.getNumGenes()];
		int flipPoint = (int) (Math.random() * Parameters.getNumGenes());
		for (int i = 0; i < Parameters.getNumGenes(); i++)
		{
			if (i > flipPoint)
			{
				parent = !parent;
			}
			if (parent)
			{
				chromosome[i] = FirstParent.chromosome[i];
			}
			else
			{
				chromosome[i] = SecondParent.chromosome[i];
			}
		}
		child.chromosome = chromosome;
		children.add(child);
		return children;
	}

	private ArrayList<Individual> NPointCrossOver(Individual FirstParent, Individual SecondParent)
	{
		ArrayList<Individual> children = new ArrayList<>();
		Individual child = new Individual();
		boolean parent = false;
		double[] chromosome = new double[Parameters.getNumGenes()];
		ArrayList<Integer> flipPoint = new ArrayList<Integer>();

		for (int i = 0; i < Parameters.numPoints; i++)
		{
			flipPoint.add((int) (Math.random() * Parameters.getNumGenes()));
		}

		for (int i = 0; i < Parameters.getNumGenes(); i++)
		{
			for (int j = 0; j < flipPoint.size(); j++)
			{
				if (i > flipPoint.get(j))
				{
					parent = !parent;
					flipPoint.remove(j);
				}
			}
			if (parent)
			{
				chromosome[i] = FirstParent.chromosome[i];
			}
			else
			{
				chromosome[i] = SecondParent.chromosome[i];
			}
		}
		child.chromosome = chromosome;
		children.add(child);
		return children;
	}

	private ArrayList<Individual> reproduce(Individual FirstParent, Individual SecondParent)
	{
		ArrayList<Individual> children = new ArrayList<>();
		switch (CrossOver)
		{
			case Clone:
				children.add(FirstParent.copy());
				children.add(SecondParent.copy());
				break;
			case Uniform:
				children = UniformCrossOver(FirstParent, SecondParent);
				break;
			case OnePoint:
				children = OnePointCrossOver(FirstParent, SecondParent);
				break;
			case NPoint:
				children = NPointCrossOver(FirstParent, SecondParent);
				break;
		}
		return children;
	}

	/**
	 * Mutation
	 * 
	 * 
	 */
	private void mutate(ArrayList<Individual> individuals)
	{
		for (Individual individual : individuals)
		{
			for (int i = 0; i < individual.chromosome.length; i++)
			{
				if (Parameters.random.nextDouble() < Parameters.mutateRate)
				{
					if (Parameters.random.nextBoolean())
					{
						individual.chromosome[i] += (Parameters.mutateChange);
					}
					else
					{
						individual.chromosome[i] -= (Parameters.mutateChange);
					}
				}
			}
		}
	}

	private void randomReplace(ArrayList<Individual> individuals)
	{
		for (int i = 0; i < Parameters.numReplacements; i++)
		{
			int randParent = (int) (Math.random() * population.size());
			int randChild = (int) (Math.random() * individuals.size());
			population.set(randParent, individuals.get(randChild));
		}
	}

	private void tournamentReplace(ArrayList<Individual> individuals)
	{
		ArrayList<Individual> newPop = new ArrayList<Individual>();
		for (int n = 0; n < Parameters.popSize; n++)
		{
			ArrayList<Individual> combinedPop = population;

			for (int i = 0; i < individuals.size(); i++)
			{
				combinedPop.add(individuals.get(i));
			}
			ArrayList<Integer> indexArray = new ArrayList<Integer>();
			for (int i = 0; i < combinedPop.size(); i++)
			{
				indexArray.add(i);
			}

			int winner = indexArray.get((int) (Math.random() * indexArray.size()));

			indexArray.remove((int) (Math.random() * indexArray.size()));

			for (int i = 0; i < Parameters.rtSize; i++)
			{

				int randomId = indexArray.get((int) (Math.random() * indexArray.size()));

				indexArray.remove((int) (Math.random() * indexArray.size()));

				winner = Tournament(winner, randomId);

				if (combinedPop.get(winner).compareTo(combinedPop.get(randomId)) != 1)
					winner = randomId;
			}
			Individual newIndividual = combinedPop.get(winner);
			newPop.add(newIndividual);
		}
		population = newPop;
	}

	private void replaceWorst(ArrayList<Individual> individuals)
	{
		for (Individual individual : individuals)
		{
			int idx = getWorstIndex();
			population.set(idx, individual);
		}
	}

	private void replace(ArrayList<Individual> individuals)
	{
		switch (Replace)
		{
			case Random:
				randomReplace(individuals);
				break;
			case Tournament:
				tournamentReplace(individuals);
				break;
			case Worst:
				replaceWorst(individuals);
				break;
		}
	}

	/**
	 * Returns the index of the worst member of the population
	 * 
	 * @return
	 */
	private int getWorstIndex()
	{
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < population.size(); i++)
		{
			Individual individual = population.get(i);
			if (worst == null)
			{
				worst = individual;
				idx = i;
			}
			else if (individual.fitness > worst.fitness)
			{
				worst = individual;
				idx = i;
			}
		}
		return idx;
	}

	@Override
	public double activationFunction(double x)
	{
		if (x < -20.0)
		{
			return -1.0;
		}
		else if (x > 20.0)
		{
			return 1.0;
		}
		return Math.tanh(x);
	}
}
