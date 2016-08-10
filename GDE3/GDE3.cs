#region License Information
/* HeuristicLab
 * Copyright (C) 2002-2016 Heuristic and Evolutionary Algorithms Laboratory (HEAL)
 *
 * This file is part of HeuristicLab.
 *
 * Implementation based on the GDE3 implementation in jMetal Framework https://github.com/jMetal/jMetal
 *
 * HeuristicLab is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HeuristicLab is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HeuristicLab. If not, see <http://www.gnu.org/licenses/>.
 */
#endregion
using System;
using System.Linq;
using System.Collections.Generic;
using HeuristicLab.Analysis;
using HeuristicLab.Common;
using HeuristicLab.Core;
using HeuristicLab.Data;
using HeuristicLab.Encodings.RealVectorEncoding;
using HeuristicLab.Operators;
using HeuristicLab.Optimization;
using HeuristicLab.Optimization.Operators;
using HeuristicLab.Parameters;
using HeuristicLab.Persistence.Default.CompositeSerializers.Storable;
using HeuristicLab.PluginInfrastructure;
using HeuristicLab.Problems.MultiObjectiveTestFunctions;
using HeuristicLab.Random;
using System.Threading;
using HeuristicLab.Algorithms.GDE3;

namespace HeuristicLab.Algoritms.GDE3
{

    [Item("Generalized Differential Evolution (GDE3)", "A generalized differential evolution algorithm.")]
    [StorableClass]
    [Creatable(CreatableAttribute.Categories.PopulationBasedAlgorithms, Priority = 400)]
    public class GDE3 : BasicAlgorithm
    {
        public override Type ProblemType
        {
            get { return typeof(MultiObjectiveTestFunctionProblem); }
        }
        public new MultiObjectiveTestFunctionProblem Problem
        {
            get { return (MultiObjectiveTestFunctionProblem)base.Problem; }
            set { base.Problem = value; }
        }

        public ILookupParameter<DoubleMatrix> BestKnownFrontParameter
        {
            get
            {
                return (ILookupParameter<DoubleMatrix>)Parameters["BestKnownFront"];
            }
        }

        private readonly IRandom _random = new MersenneTwister();
        private int evals;
        private double IGDSumm;

        #region ParameterNames
        private const string MaximumGenerationsParameterName = "Maximum Generations";
        private const string MaximumEvaluationsParameterName = "Maximum Evaluations";
        private const string CrossoverProbabilityParameterName = "CrossoverProbability";
        private const string PopulationSizeParameterName = "PopulationSize";
        private const string ScalingFactorParameterName = "ScalingFactor";

        #endregion

        #region ParameterProperties
        public IFixedValueParameter<IntValue> MaximumGenerationsParameter
        {
            get { return (IFixedValueParameter<IntValue>)Parameters[MaximumGenerationsParameterName]; }
        }
        public IFixedValueParameter<IntValue> MaximumEvaluationsParameter
        {
            get { return (IFixedValueParameter<IntValue>)Parameters[MaximumEvaluationsParameterName]; }
        }
        private ValueParameter<IntValue> PopulationSizeParameter
        {
            get { return (ValueParameter<IntValue>)Parameters[PopulationSizeParameterName]; }
        }
        public ValueParameter<DoubleValue> CrossoverProbabilityParameter
        {
            get { return (ValueParameter<DoubleValue>)Parameters[CrossoverProbabilityParameterName]; }
        }
        public ValueParameter<DoubleValue> ScalingFactorParameter
        {
            get { return (ValueParameter<DoubleValue>)Parameters[ScalingFactorParameterName]; }
        }
        #endregion

        #region Properties
        public int MaximumGenerations
        {
            get { return MaximumGenerationsParameter.Value.Value; }
            set { MaximumGenerationsParameter.Value.Value = value; }
        }

        public int MaximumEvaluations
        {
            get { return MaximumEvaluationsParameter.Value.Value; }
            set { MaximumEvaluationsParameter.Value.Value = value; }
        }

        public Double CrossoverProbability
        {
            get { return CrossoverProbabilityParameter.Value.Value; }
            set { CrossoverProbabilityParameter.Value.Value = value; }
        }
        public Double ScalingFactor
        {
            get { return ScalingFactorParameter.Value.Value; }
            set { ScalingFactorParameter.Value.Value = value; }
        }
        public IntValue PopulationSize
        {
            get { return PopulationSizeParameter.Value; }
            set { PopulationSizeParameter.Value = value; }
        }
        #endregion

        #region ResultsProperties
        private double ResultsBestQuality
        {
            get { return ((DoubleValue)Results["Best Quality"].Value).Value; }
            set { ((DoubleValue)Results["Best Quality"].Value).Value = value; }
        }

        private double ResultsIGDMean
        {
            get { return ((DoubleValue)Results["IGDMeanValue"].Value).Value; }
            set { ((DoubleValue)Results["IGDMeanValue"].Value).Value = value; }
        }

        private double ResultsIGDBest
        {
            get { return ((DoubleValue)Results["IGDBestValue"].Value).Value; }
            set { ((DoubleValue)Results["IGDBestValue"].Value).Value = value; }
        }

        private double ResultsIGDWorst
        {
            get { return ((DoubleValue)Results["IGDWorstValue"].Value).Value; }
            set { ((DoubleValue)Results["IGDWorstValue"].Value).Value = value; }
        }

        private double ResultsInvertedGenerationalDistance
        {
            get { return ((DoubleValue)Results["InvertedGenerationalDistance"].Value).Value; }
            set { ((DoubleValue)Results["InvertedGenerationalDistance"].Value).Value = value; }
        }

        private double ResultsHypervolume
        {
            get { return ((DoubleValue)Results["HyperVolumeValue"].Value).Value; }
            set { ((DoubleValue)Results["HyperVolumeValue"].Value).Value = value; }
        }

        private DoubleMatrix ResultsBestFront
        {
            get { return (DoubleMatrix)Results["Best Front"].Value; }
            set { Results["Best Front"].Value = value; }
        }

        private int ResultsEvaluations
        {
            get { return ((IntValue)Results["Evaluations"].Value).Value; }
            set { ((IntValue)Results["Evaluations"].Value).Value = value; }
        }
        private int ResultsGenerations
        {
            get { return ((IntValue)Results["Generations"].Value).Value; }
            set { ((IntValue)Results["Generations"].Value).Value = value; }
        }
        private double ResultsGenerationalDistance
        {
            get { return ((DoubleValue)Results["GenerationalDistance"].Value).Value; }
            set { ((DoubleValue)Results["GenerationalDistance"].Value).Value = value; }
        }

        private double ResultsSpacing
        {
            get { return ((DoubleValue)Results["Spacing"].Value).Value; }
            set { ((DoubleValue)Results["Spacing"].Value).Value = value; }
        }

        private double ResultsCrowding
        {
            get { return ((DoubleValue)Results["Crowding"].Value).Value; }
            set { ((DoubleValue)Results["Crowding"].Value).Value = value; }
        }

        #endregion

        [StorableConstructor]
        protected GDE3(bool deserializing) : base(deserializing) { }

        protected GDE3(GDE3 original, Cloner cloner)
          : base(original, cloner)
        {
        }

        public override IDeepCloneable Clone(Cloner cloner)
        {
            return new GDE3(this, cloner);
        }

        public GDE3()
        {
            Parameters.Add(new FixedValueParameter<IntValue>(MaximumGenerationsParameterName, "", new IntValue(1000)));
            Parameters.Add(new FixedValueParameter<IntValue>(MaximumEvaluationsParameterName, "", new IntValue(Int32.MaxValue)));
            Parameters.Add(new ValueParameter<IntValue>(PopulationSizeParameterName, "The size of the population of solutions.", new IntValue(100)));
            Parameters.Add(new ValueParameter<DoubleValue>(CrossoverProbabilityParameterName, "The value for crossover rate", new DoubleValue(0.5)));
            Parameters.Add(new ValueParameter<DoubleValue>(ScalingFactorParameterName, "The value for scaling factor", new DoubleValue(0.5)));
            Parameters.Add(new LookupParameter<DoubleMatrix>("BestKnownFront", "The currently best known Pareto front"));
        }

        protected override void Run(CancellationToken cancellationToken)
        {
            // Set up the results display
            Results.Add(new Result("Generations", new IntValue(0)));
            Results.Add(new Result("Evaluations", new IntValue(0)));
            Results.Add(new Result("Best Front", new DoubleMatrix()));
            Results.Add(new Result("Crowding", new DoubleValue(0)));
            Results.Add(new Result("InvertedGenerationalDistance", new DoubleValue(0)));
            Results.Add(new Result("GenerationalDistance", new DoubleValue(0)));
            Results.Add(new Result("HyperVolumeValue", new DoubleValue(0)));
            Results.Add(new Result("IGDMeanValue", new DoubleValue(0)));
            Results.Add(new Result("IGDBestValue", new DoubleValue(Int32.MaxValue)));
            Results.Add(new Result("IGDWorstValue", new DoubleValue(0)));

            Results.Add(new Result("Spacing", new DoubleValue(0)));
            Results.Add(new Result("Scatterplot", typeof(IMOFrontModel)));
            var table = new DataTable("Qualities");
            table.Rows.Add(new DataRow("Best Quality"));
            Results.Add(new Result("Qualities", table));

            //setup the variables
            List<SolutionSet> population;
            List<SolutionSet> offspringPopulation;
            SolutionSet[] parent;
            double IGDSumm = 0;
            
            //initialize population
            population = new List<SolutionSet>(PopulationSizeParameter.Value.Value);

            for (int i = 0; i < PopulationSizeParameter.Value.Value; ++i)
            {
                var m = createIndividual();
                m.Quality = Problem.Evaluate(m.Population, _random);
                //the test function is constrained
                if (m.Quality.Length > Problem.Objectives)
                {
                    m.OverallConstrainViolation = m.Quality[Problem.Objectives];
                } else {
                    m.OverallConstrainViolation = 0;
                }
                population.Add(m);
            }

            this.initProgress();
            int generations = 1;

            while (ResultsEvaluations < MaximumEvaluationsParameter.Value.Value
               && !cancellationToken.IsCancellationRequested)
            {
                var populationSize = PopulationSizeParameter.Value.Value;

                // Create the offSpring solutionSet
                offspringPopulation = new List<SolutionSet>(PopulationSizeParameter.Value.Value * 2);

                for (int i = 0; i < populationSize; i++)
                {
                    // Obtain parents. Two parameters are required: the population and the 
                    //                 index of the current individual
                    parent = selection(population, i);

                    SolutionSet child;
                    // Crossover. The parameters are the current individual and the index of the array of parents 
                    child = reproduction(population[i], parent);

                    child.Quality = Problem.Evaluate(child.Population, _random);

                    this.updateProgres();

                    //the test function is constrained
                    if (child.Quality.Length > Problem.Objectives)
                    {
                        child.OverallConstrainViolation = child.Quality[Problem.Objectives];
                    } else {
                        child.OverallConstrainViolation = 0;
                    }

                    // Dominance test
                    int result;
                    result = compareDomination(population[i], child);

                    if (result == -1)
                    { // Solution i dominates child
                        offspringPopulation.Add(population[i]);
                    }
                    else if (result == 1)
                    { // child dominates
                        offspringPopulation.Add(child);
                    }
                    else
                    { // the two solutions are non-dominated
                        offspringPopulation.Add(child);
                        offspringPopulation.Add(population[i]);
                    }
                }

                // Ranking the offspring population
                List<SolutionSet>[] ranking = computeRanking(offspringPopulation);
                population = crowdingDistanceSelection(ranking);
                generations++;
                ResultsGenerations = generations;
                displayResults(population);
            }
        }

        private void displayResults(List<SolutionSet> population)
        {
            List<SolutionSet>[] rankingFinal = computeRanking(population);

            int objectives = Problem.Objectives;
            var optimalfront = Problem.TestFunction.OptimalParetoFront(objectives);

            double[][] opf = new double[0][];
            if (optimalfront != null)
            {
                opf = optimalfront.Select(s => s.ToArray()).ToArray();
            }

            //compute the final qualities and population
            double[][] qualitiesFinal = new double[rankingFinal[0].Count][];
            double[][] populationFinal = new double[rankingFinal[0].Count][];

            for (int i = 0; i < rankingFinal[0].Count; ++i)
            {
                qualitiesFinal[i] = new double[Problem.Objectives];
                populationFinal[i] = new double[Problem.Objectives];
                for (int j = 0; j < Problem.Objectives; ++j)
                {
                    populationFinal[i][j] = rankingFinal[0][i].Population[j];
                    qualitiesFinal[i][j] = rankingFinal[0][i].Quality[j];
                }
            }
            IEnumerable<double[]> en = qualitiesFinal;
            IEnumerable<double[]> frontVectors = NonDominatedSelect.selectNonDominatedVectors(qualitiesFinal, Problem.TestFunction.Maximization(objectives), true);
            //update the results

            ResultsEvaluations = this.evals;
            ResultsBestFront = new DoubleMatrix(MultiObjectiveTestFunctionProblem.To2D(qualitiesFinal));
            ResultsCrowding = Crowding.Calculate(qualitiesFinal, Problem.TestFunction.Bounds(objectives));
            GenerationalDistanceCalculator distance = new GenerationalDistanceCalculator();
            ResultsInvertedGenerationalDistance = distance.CalculateGenerationalDistance(qualitiesFinal, opf, Problem.Objectives);
            ResultsHypervolume = Hypervolume.Calculate(frontVectors, Problem.TestFunction.ReferencePoint(objectives), Problem.TestFunction.Maximization(objectives));
            ResultsGenerationalDistance = GenerationalDistance.Calculate(qualitiesFinal, optimalfront, 1);
            Results["Scatterplot"].Value = new MOSolution(qualitiesFinal, populationFinal, opf, objectives);
            ResultsSpacing = Spacing.Calculate(qualitiesFinal);

            if (ResultsIGDBest > ResultsInvertedGenerationalDistance) {
                ResultsIGDBest = ResultsInvertedGenerationalDistance;
            }
            if (ResultsIGDWorst < ResultsInvertedGenerationalDistance)
            {
                ResultsIGDWorst = ResultsInvertedGenerationalDistance;
            }
            this.IGDSumm += ResultsInvertedGenerationalDistance;
            ResultsIGDMean = this.IGDSumm / ResultsGenerations;
        }

        private int getWorstIndex(List<SolutionSet> SolutionsList)
        {
            int result = 0;

            if ((SolutionsList == null) || SolutionsList.Count == 0)
            {
                result = 0;
            }
            else
            {
                SolutionSet worstKnown = SolutionsList[0],
                            candidateSolution;
                int flag;
                for (int i = 1; i < SolutionsList.Count; i++)
                {
                    candidateSolution = SolutionsList[i];
                    flag = compareDomination(worstKnown, candidateSolution);
                    if (flag == -1)
                    {
                        result = i;
                        worstKnown = candidateSolution;
                    }
                }
            }
            return result;
        }

        private SolutionSet createIndividual()
        {
            var dim = Problem.ProblemSize;
            var lb = Problem.Bounds[0, 0];
            var ub = Problem.Bounds[0, 1];
            var range = ub - lb;
            var v = new double[Problem.ProblemSize];
            SolutionSet solutionObject = new SolutionSet(PopulationSizeParameter.Value.Value);

            for (int i = 0; i < Problem.ProblemSize; ++i)
            {
                v[i] = _random.NextDouble() * range + lb;

            }
            solutionObject.createSolution(v);
            return solutionObject;
        }

        private SolutionSet createEmptyIndividual()
        {
            SolutionSet solutionObject = new SolutionSet(PopulationSizeParameter.Value.Value);
            var n = new RealVector(Problem.ProblemSize);
            solutionObject.Population = n;
            return solutionObject;
        }

        private void initProgress()
        {
            this.evals = PopulationSizeParameter.Value.Value;
        }

        private void updateProgres()
        {
            this.evals++;
        }

        private SolutionSet[] selection(List<SolutionSet> population, int i)
        {
            SolutionSet[] parents = new SolutionSet[3];
            int r0, r1, r2;
            //assure the selected vectors r0, r1 and r2 are different
            do
            {
                r0 = _random.Next(0, PopulationSizeParameter.Value.Value);
            } while (r0 == i);
            do
            {
                r1 = _random.Next(0, PopulationSizeParameter.Value.Value);
            } while (r1 == i || r1 == r0);
            do
            {
                r2 = _random.Next(0, PopulationSizeParameter.Value.Value);
            } while (r2 == i || r2 == r0 || r2 == r1);

            parents[0] = population[r0];
            parents[1] = population[r1];
            parents[2] = population[r2];

            return parents;
        }

        private SolutionSet reproduction(SolutionSet parent, SolutionSet[] parentsSolutions)
        {
            var individual = createEmptyIndividual();
            double rnbr = _random.Next(0, Problem.ProblemSize);
            for (int m = 0; m < Problem.ProblemSize; m++)
            {
                if (_random.NextDouble() < CrossoverProbabilityParameter.Value.Value || m == rnbr)
                {
                    double value;
                    value = parentsSolutions[2].Population[m] +
                        ScalingFactorParameter.Value.Value * (parentsSolutions[0].Population[m] - parentsSolutions[1].Population[m]);
                    //check the problem upper and lower bounds
                    if (value > Problem.Bounds[0, 1]) value = Problem.Bounds[0, 1];
                    if (value < Problem.Bounds[0, 0]) value = Problem.Bounds[0, 0];
                    individual.Population[m] = value;
                }
                else
                {
                    double value;
                    value = parent.Population[m];
                    individual.Population[m] = value;
                }
            }
            return individual;
        }

        private List<SolutionSet> crowdingDistanceSelection(List<SolutionSet>[] ranking)
        {
            List<SolutionSet> population = new List<SolutionSet>();
            int rankingIndex = 0;
            while (populationIsNotFull(population))
            {
                if (subFrontFillsIntoThePopulation(ranking, rankingIndex, population))
                {
                    addRankedSolutionToPopulation(ranking, rankingIndex, population);
                    rankingIndex++;
                }
                else {
                    crowdingDistanceAssignment(ranking[rankingIndex]);
                    addLastRankedSolutionToPopulation(ranking, rankingIndex, population);
                }
            }
            return population;
        }

        private void addLastRankedSolutionToPopulation(List<SolutionSet>[] ranking, int rankingIndex, List<SolutionSet> population)
        {
            List<SolutionSet> currentRankedFront = ranking[rankingIndex];
            //descending sort and add the front with highest crowding distance to the population
            currentRankedFront.Sort((x, y) => -x.CrowdingDistance.CompareTo(y.CrowdingDistance));
            int i = 0;
            while (population.Count < PopulationSizeParameter.Value.Value)
            {
                population.Add(currentRankedFront[i]);
                i++;
            }
        }

        private void crowdingDistanceAssignment(List<SolutionSet> rankingSubfront)
        {
            int size = rankingSubfront.Count;

            if (size == 0)
                return;

            if (size == 1)
            {
                rankingSubfront[0].CrowdingDistance = double.PositiveInfinity;
                return;
            }

            if (size == 2)
            {
                rankingSubfront[0].CrowdingDistance = double.PositiveInfinity;
                rankingSubfront[1].CrowdingDistance = double.PositiveInfinity;
                return;
            }

            //Use a new SolutionSet to evite alter original solutionSet
            List<SolutionSet> front = new List<SolutionSet>(size);
            for (int i = 0; i < size; i++)
            {
                front.Add(rankingSubfront[i]);
            }

            for (int i = 0; i < size; i++)
                front[i].CrowdingDistance = 0.0;

            double objetiveMaxn;
            double objetiveMinn;
            double distance;

            for (int i = 0; i < Problem.Objectives; i++)
            {
                // Sort the front population by the objective i          
                front.Sort((x, y) => x.Quality[i].CompareTo(y.Quality[i]));
                objetiveMinn = front[0].Quality[i];
                objetiveMaxn = front[front.Count - 1].Quality[i];

                //Set crowding distance for the current front           
                front[0].CrowdingDistance = double.PositiveInfinity;
                front[size - 1].CrowdingDistance = double.PositiveInfinity;

                for (int j = 1; j < size - 1; j++)
                {
                    distance = front[j + 1].Quality[i] - front[j - 1].Quality[i];
                    distance = distance / (objetiveMaxn - objetiveMinn);
                    distance += front[j].CrowdingDistance;
                    front[j].CrowdingDistance = distance;
                }
            }
        }

        private void addRankedSolutionToPopulation(List<SolutionSet>[] ranking, int rankingIndex, List<SolutionSet> population)
        {
            foreach (SolutionSet solution in ranking[rankingIndex])
            {
                population.Add(solution);
            }
        }

        private bool subFrontFillsIntoThePopulation(List<SolutionSet>[] ranking, int rankingIndex, List<SolutionSet> population)
        {
            return ranking[rankingIndex].Count < (PopulationSizeParameter.Value.Value - population.Count);
        }

        private bool populationIsNotFull(List<SolutionSet> population)
        {
            return population.Count < PopulationSizeParameter.Value.Value;
        }

        private List<SolutionSet>[] computeRanking(List<SolutionSet> tmpList)
        {
            // dominateMe[i] contains the number of solutions dominating i        
            int[] dominateMe = new int[tmpList.Count];

            // iDominate[k] contains the list of solutions dominated by k
            List<int>[] iDominate = new List<int>[tmpList.Count];

            // front[i] contains the list of individuals belonging to the front i
            List<int>[] front = new List<int>[tmpList.Count + 1];

            // flagDominate is an auxiliar encodings.variable
            int flagDominate;

            // Initialize the fronts 
            for (int i = 0; i < front.Length; i++)
            {
                front[i] = new List<int>();
            }

            //-> Fast non dominated sorting algorithm
            // Contribution of Guillaume Jacquenot
            for (int p = 0; p < tmpList.Count; p++)
            {
                // Initialize the list of individuals that i dominate and the number
                // of individuals that dominate me
                iDominate[p] = new List<int>(); 
                dominateMe[p] = 0;
            }
            for (int p = 0; p < (tmpList.Count - 1); p++)
            {
                // For all q individuals , calculate if p dominates q or vice versa
                for (int q = p + 1; q < tmpList.Count; q++)
                {
                    flagDominate = compareConstraintsViolation(tmpList[p], tmpList[q]);
                    if (flagDominate == 0) { 
                        flagDominate = compareDomination(tmpList[p], tmpList[q]);
                    }
                    if (flagDominate == -1)
                    {
                        iDominate[p].Add(q);
                        dominateMe[q]++;
                    }
                    else if (flagDominate == 1)
                    {
                        iDominate[q].Add(p);
                        dominateMe[p]++;
                    }
                }
                // If nobody dominates p, p belongs to the first front
            }
            for (int i = 0; i < tmpList.Count; i++)
            {
                if (dominateMe[i] == 0)
                {
                    front[0].Add(i);
                    tmpList[i].Rank = 0;
                }
            }

            //Obtain the rest of fronts
            int k = 0;

            while (front[k].Count != 0)
            {
                k++;
                foreach (var it1 in front[k - 1])
                {
                    foreach (var it2 in iDominate[it1])
                    {
                        int index = it2;
                        dominateMe[index]--;
                        if (dominateMe[index] == 0)
                        {
                            front[k].Add(index);
                            tmpList[index].Rank = k;
                        }
                    }
                }
            }
            //<-

            var rankedSubpopulation = new List<SolutionSet>[k];
            //0,1,2,....,i-1 are front, then i fronts
            for (int j = 0; j < k; j++)
            {
                rankedSubpopulation[j] = new List<SolutionSet>(front[j].Count);
                foreach (var it1 in front[j])
                {
                    rankedSubpopulation[j].Add(tmpList[it1]);
                }
            }
            return rankedSubpopulation;
        }

        private int compareDomination(SolutionSet solution1, SolutionSet solution2)
        {
            int dominate1; // dominate1 indicates if some objective of solution1 
                           // dominates the same objective in solution2. dominate2
            int dominate2; // is the complementary of dominate1.

            dominate1 = 0;
            dominate2 = 0;

            int flag; //stores the result of the comparison

            // Test to determine whether at least a solution violates some constraint
            if (needToCompareViolations(solution1, solution2))
            {
                return compareConstraintsViolation(solution1, solution2);
            }

            // Equal number of violated constraints. Applying a dominance Test then
            double value1, value2;
            for (int i = 0; i < Problem.Objectives; i++)
            {
                value1 = solution1.Quality[i];
                value2 = solution2.Quality[i];
                if (value1 < value2)
                {
                    flag = -1;
                }
                else if (value2 < value1)
                {
                    flag = 1;
                }
                else
                {
                    flag = 0;
                }

                if (flag == -1)
                {
                    dominate1 = 1;
                }

                if (flag == 1)
                {
                    dominate2 = 1;
                }
            }

            if (dominate1 == dominate2)
            {
                return 0; //No one dominate the other
            }
            if (dominate1 == 1)
            {
                return -1; // solution1 dominate
            }
            return 1;    // solution2 dominate   
        }

        private bool needToCompareViolations(SolutionSet solution1, SolutionSet solution2)
        {
            bool needToCompare;
            needToCompare = (solution1.OverallConstrainViolation < 0) || (solution2.OverallConstrainViolation < 0);

            return needToCompare;
        }

        private int compareConstraintsViolation(SolutionSet solution1, SolutionSet solution2)
        {
            int result;
            double overall1, overall2;
            overall1 = solution1.OverallConstrainViolation;
            overall2 = solution2.OverallConstrainViolation;

            if ((overall1 < 0) && (overall2 < 0))
            {
                if (overall1 > overall2)
                {
                    result = -1;
                }
                else if (overall2 > overall1)
                {
                    result = 1;
                }
                else
                {
                    result = 0;
                }
            }
            else if ((overall1 == 0) && (overall2 < 0))
            {
                result = -1;
            }
            else if ((overall1 < 0) && (overall2 == 0))
            {
                result = 1;
            }
            else
            {
                result = 0;
            }
            return result;
        }
    }
}



