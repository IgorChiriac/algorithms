#region License Information
/* HeuristicLab
 * Copyright (C) 2002-2016 Heuristic and Evolutionary Algorithms Laboratory (HEAL)
 *
 * This file is part of HeuristicLab.
 *
 * The implementation is inspired by the implementation in JAVA of SHADE algorithm https://sites.google.com/site/tanaberyoji/software/SHADE1.0.1_CEC2013.zip?attredirects=0&d=1
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
using HeuristicLab.Analysis;
using HeuristicLab.Common;
using HeuristicLab.Core;
using HeuristicLab.Data;
using HeuristicLab.Encodings.RealVectorEncoding;
using HeuristicLab.Optimization;
using HeuristicLab.Parameters;
using HeuristicLab.Persistence.Default.CompositeSerializers.Storable;
using HeuristicLab.Problems.TestFunctions;
using HeuristicLab.Random;
using System;
using System.Collections.Generic;
using System.Threading;

namespace HeuristicLab.Algorithms.Shade
{

    [Item("Success-History Based Parameter Adaptation for DE (SHADE)", "A self-adaptive version of differential evolution")]
    [StorableClass]
    [Creatable(CreatableAttribute.Categories.PopulationBasedAlgorithms, Priority = 400)]
    public class Shade : BasicAlgorithm
    {
        public Func<IEnumerable<double>, double> Evaluation;

        public override Type ProblemType
        {
            get { return typeof(SingleObjectiveTestFunctionProblem); }
        }
        public new SingleObjectiveTestFunctionProblem Problem
        {
            get { return (SingleObjectiveTestFunctionProblem)base.Problem; }
            set { base.Problem = value; }
        }

        private readonly IRandom _random = new MersenneTwister();
        private int evals;
        private int pop_size;
        private double arc_rate;
        private int arc_size;
        private double p_best_rate;
        private int memory_size;

        private double[][] pop;
        private double[] fitness;
        private double[][] children;
        private double[] children_fitness;

        private double[] bsf_solution;
        private double bsf_fitness = 1e+30;
        private double[,] archive;
        private int num_arc_inds = 0;

        #region ParameterNames
        private const string MaximumEvaluationsParameterName = "Maximum Evaluations";
        private const string SeedParameterName = "Seed";
        private const string SetSeedRandomlyParameterName = "SetSeedRandomly";
        private const string CrossoverProbabilityParameterName = "CrossoverProbability";
        private const string PopulationSizeParameterName = "PopulationSize";
        private const string ScalingFactorParameterName = "ScalingFactor";
        private const string ValueToReachParameterName = "ValueToReach";
        private const string ArchiveRateParameterName = "ArchiveRate";
        private const string MemorySizeParameterName = "MemorySize";
        private const string BestRateParameterName = "BestRate";
        #endregion

        #region ParameterProperties
        public IFixedValueParameter<IntValue> MaximumEvaluationsParameter
        {
            get { return (IFixedValueParameter<IntValue>)Parameters[MaximumEvaluationsParameterName]; }
        }
        public IFixedValueParameter<IntValue> SeedParameter
        {
            get { return (IFixedValueParameter<IntValue>)Parameters[SeedParameterName]; }
        }
        public FixedValueParameter<BoolValue> SetSeedRandomlyParameter
        {
            get { return (FixedValueParameter<BoolValue>)Parameters[SetSeedRandomlyParameterName]; }
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
        public ValueParameter<DoubleValue> ValueToReachParameter
        {
            get { return (ValueParameter<DoubleValue>)Parameters[ValueToReachParameterName]; }
        }
        public ValueParameter<DoubleValue> ArchiveRateParameter
        {
            get { return (ValueParameter<DoubleValue>)Parameters[ArchiveRateParameterName]; }
        }
        public ValueParameter<IntValue> MemorySizeParameter
        {
            get { return (ValueParameter<IntValue>)Parameters[MemorySizeParameterName]; }
        }
        public ValueParameter<DoubleValue> BestRateParameter
        {
            get { return (ValueParameter<DoubleValue>)Parameters[BestRateParameterName]; }
        }
        #endregion

        #region Properties
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
        public int Seed
        {
            get { return SeedParameter.Value.Value; }
            set { SeedParameter.Value.Value = value; }
        }
        public bool SetSeedRandomly
        {
            get { return SetSeedRandomlyParameter.Value.Value; }
            set { SetSeedRandomlyParameter.Value.Value = value; }
        }
        public IntValue PopulationSize
        {
            get { return PopulationSizeParameter.Value; }
            set { PopulationSizeParameter.Value = value; }
        }
        public Double ValueToReach
        {
            get { return ValueToReachParameter.Value.Value; }
            set { ValueToReachParameter.Value.Value = value; }
        }
        public Double ArchiveRate
        {
            get { return ArchiveRateParameter.Value.Value; }
            set { ArchiveRateParameter.Value.Value = value; }
        }
        public IntValue MemorySize
        {
            get { return MemorySizeParameter.Value; }
            set { MemorySizeParameter.Value = value; }
        }
        public Double BestRate
        {
            get { return BestRateParameter.Value.Value; }
            set { BestRateParameter.Value.Value = value; }
        }
        #endregion

        #region ResultsProperties
        private double ResultsBestQuality
        {
            get { return ((DoubleValue)Results["Best Quality"].Value).Value; }
            set { ((DoubleValue)Results["Best Quality"].Value).Value = value; }
        }

        private double VTRBestQuality
        {
            get { return ((DoubleValue)Results["VTR"].Value).Value; }
            set { ((DoubleValue)Results["VTR"].Value).Value = value; }
        }

        private RealVector ResultsBestSolution
        {
            get { return (RealVector)Results["Best Solution"].Value; }
            set { Results["Best Solution"].Value = value; }
        }

        private int ResultsEvaluations
        {
            get { return ((IntValue)Results["Evaluations"].Value).Value; }
            set { ((IntValue)Results["Evaluations"].Value).Value = value; }
        }
        private int ResultsIterations
        {
            get { return ((IntValue)Results["Iterations"].Value).Value; }
            set { ((IntValue)Results["Iterations"].Value).Value = value; }
        }

        private DataTable ResultsQualities
        {
            get { return ((DataTable)Results["Qualities"].Value); }
        }
        private DataRow ResultsQualitiesBest
        {
            get { return ResultsQualities.Rows["Best Quality"]; }
        }

        #endregion

        [StorableConstructor]
        protected Shade(bool deserializing) : base(deserializing) { }

        protected Shade(Shade original, Cloner cloner)
          : base(original, cloner)
        {
        }

        public override IDeepCloneable Clone(Cloner cloner)
        {
            return new Shade(this, cloner);
        }

        public Shade()
        {
            Parameters.Add(new FixedValueParameter<IntValue>(MaximumEvaluationsParameterName, "", new IntValue(Int32.MaxValue)));
            Parameters.Add(new ValueParameter<IntValue>(PopulationSizeParameterName, "The size of the population of solutions.", new IntValue(75)));
            Parameters.Add(new ValueParameter<DoubleValue>(ValueToReachParameterName, "Value to reach (VTR) parameter", new DoubleValue(0.00000001)));
            Parameters.Add(new ValueParameter<DoubleValue>(ArchiveRateParameterName, "Archive rate parameter", new DoubleValue(2.0)));
            Parameters.Add(new ValueParameter<IntValue>(MemorySizeParameterName, "Memory size parameter", new IntValue(0)));
            Parameters.Add(new ValueParameter<DoubleValue>(BestRateParameterName, "Best rate parameter", new DoubleValue(0.1)));
        }

        protected override void Run(CancellationToken cancellationToken)
        {

            // Set up the results display
            Results.Add(new Result("Iterations", new IntValue(0)));
            Results.Add(new Result("Evaluations", new IntValue(0)));
            Results.Add(new Result("Best Solution", new RealVector()));
            Results.Add(new Result("Best Quality", new DoubleValue(double.NaN)));
            Results.Add(new Result("VTR", new DoubleValue(double.NaN)));
            var table = new DataTable("Qualities");
            table.Rows.Add(new DataRow("Best Quality"));
            Results.Add(new Result("Qualities", table));


            this.evals = 0;
            int archive_size = (int)Math.Round(ArchiveRateParameter.Value.Value * PopulationSize.Value);
            int problem_size = Problem.ProblemSize.Value;

            int pop_size = PopulationSizeParameter.Value.Value;
            this.arc_rate = ArchiveRateParameter.Value.Value;
            this.arc_size = (int)Math.Round(this.arc_rate * pop_size);
            this.p_best_rate = BestRateParameter.Value.Value;
            this.memory_size = MemorySizeParameter.Value.Value;

            this.pop = new double[pop_size][];
            this.fitness = new double[pop_size];
            this.children = new double[pop_size][];
            this.children_fitness = new double[pop_size];

            this.bsf_solution = new double[problem_size];
            this.bsf_fitness = 1e+30;
            this.archive = new double[arc_size, Problem.ProblemSize.Value];
            this.num_arc_inds = 0;

            double[,] populationOld = new double[PopulationSizeParameter.Value.Value, Problem.ProblemSize.Value];
            double[,] mutationPopulation = new double[PopulationSizeParameter.Value.Value, Problem.ProblemSize.Value];
            double[,] trialPopulation = new double[PopulationSizeParameter.Value.Value, Problem.ProblemSize.Value];
            double[] bestPopulation = new double[Problem.ProblemSize.Value];
            double[] bestPopulationIteration = new double[Problem.ProblemSize.Value];
            double[,] archive = new double[archive_size, Problem.ProblemSize.Value];


            // //for external archive
            int rand_arc_ind;

            int num_success_params;

            double[] success_sf = new double[PopulationSizeParameter.Value.Value];
            double[] success_cr = new double[PopulationSizeParameter.Value.Value];
            double[] dif_fitness = new double[PopulationSizeParameter.Value.Value];
            double[] fitness = new double[PopulationSizeParameter.Value.Value];

            // the contents of M_f and M_cr are all initialiezed 0.5
            double[] memory_sf = new double[MemorySizeParameter.Value.Value];
            double[] memory_cr = new double[MemorySizeParameter.Value.Value];

            for (int i = 0; i < MemorySizeParameter.Value.Value; i++)
            {
                memory_sf[i] = 0.5;
                memory_cr[i] = 0.5;
            }

            //memory index counter
            int memory_pos = 0;
            double temp_sum_sf1, temp_sum_sf2, temp_sum_cr1, temp_sum_cr2, temp_sum, temp_weight;

            //for new parameters sampling
            double mu_sf, mu_cr;
            int rand_mem_index;

            double[] pop_sf = new double[PopulationSizeParameter.Value.Value];
            double[] pop_cr = new double[PopulationSizeParameter.Value.Value];

            //for current-to-pbest/1
            int p_best_ind;
            double m = PopulationSizeParameter.Value.Value * BestRateParameter.Value.Value;
            int p_num = (int)Math.Round(m);
            int[] sorted_array = new int[PopulationSizeParameter.Value.Value];
            double[] sorted_fitness = new double[PopulationSizeParameter.Value.Value];

            //initialize the population
            populationOld = makeNewIndividuals();

            //evaluate the best member after the intialiazation
            //the idea is to select first member and after that to check the others members from the population

            int best_index = 0;
            double[] populationRow = new double[Problem.ProblemSize.Value];
            bestPopulation = getMatrixRow(populationOld, best_index);
            RealVector bestPopulationVector = new RealVector(bestPopulation);
            double bestPopulationValue = Obj(bestPopulationVector);
            fitness[best_index] = bestPopulationValue;
            RealVector selectionVector;
            RealVector trialVector;
            double qtrial;


            for (var i = 0; i < PopulationSizeParameter.Value.Value; i++)
            {
                populationRow = getMatrixRow(populationOld, i);
                trialVector = new RealVector(populationRow);

                qtrial = Obj(trialVector);
                fitness[i] = qtrial;

                if (qtrial > bestPopulationValue)
                {
                    bestPopulationVector = new RealVector(populationRow);
                    bestPopulationValue = qtrial;
                    best_index = i;
                }
            }

            int iterations = 1;

            // Loop until iteration limit reached or canceled.
            // todo replace with a function
            // && bestPopulationValue > Problem.BestKnownQuality.Value + ValueToReachParameter.Value.Value
            while (ResultsEvaluations < MaximumEvaluations
                && !cancellationToken.IsCancellationRequested &&
                bestPopulationValue > Problem.BestKnownQuality.Value + ValueToReachParameter.Value.Value)
            {
                for (int i = 0; i < PopulationSizeParameter.Value.Value; i++) sorted_array[i] = i;
                for (int i = 0; i < PopulationSizeParameter.Value.Value; i++) sorted_fitness[i] = fitness[i];

                Quicksort(sorted_fitness, 0, PopulationSizeParameter.Value.Value - 1, sorted_array);

                for (int target = 0; target < PopulationSizeParameter.Value.Value; target++)
                {
                    rand_mem_index = (int)(_random.NextDouble() * MemorySizeParameter.Value.Value);
                    mu_sf = memory_sf[rand_mem_index];
                    mu_cr = memory_cr[rand_mem_index];

                    //generate CR_i and repair its value
                    if (mu_cr == -1)
                    {
                        pop_cr[target] = 0;
                    }
                    else {
                        pop_cr[target] = gauss(mu_cr, 0.1);
                        if (pop_cr[target] > 1) pop_cr[target] = 1;
                        else if (pop_cr[target] < 0) pop_cr[target] = 0;
                    }

                    //generate F_i and repair its value
                    do {
                        pop_sf[target] = cauchy_g(mu_sf, 0.1);
                    } while (pop_sf[target] <= 0);

                    if (pop_sf[target] > 1) pop_sf[target] = 1;

                    //p-best individual is randomly selected from the top pop_size *  p_i members
                    p_best_ind = sorted_array[(int)(_random.NextDouble() * p_num)];

                    trialPopulation = operateCurrentToPBest1BinWithArchive(populationOld, trialPopulation, target, p_best_ind, pop_sf[target], pop_cr[target]);
                }

                for (int i = 0; i < pop_size; i++) {
                    trialVector = new RealVector(getMatrixRow(trialPopulation, i));
                    children_fitness[i] = Obj(trialVector);
                }

                //update bfs solution 
                for (var i = 0; i < PopulationSizeParameter.Value.Value; i++)
                {
                    populationRow = getMatrixRow(populationOld, i);
                    qtrial = fitness[i];

                    if (qtrial > bestPopulationValue)
                    {
                        bestPopulationVector = new RealVector(populationRow);
                        bestPopulationValue = qtrial;
                        best_index = i;
                    }
                }

                num_success_params = 0;

                //generation alternation
                for (int i = 0; i < pop_size; i++)
                {
                    if (children_fitness[i] == fitness[i])
                    {
                        fitness[i] = children_fitness[i];
                        for (int j = 0; j < problem_size; j++) populationOld[i,j] = trialPopulation[i,j];
                    }
                    else if (children_fitness[i] < fitness[i])
                    {
                        //parent vectors x_i which were worse than the trial vectors u_i are preserved
                        if (arc_size > 1)
                        {
                            if (num_arc_inds < arc_size)
                            {
                                for (int j = 0; j < problem_size; j++) this.archive[num_arc_inds, j] = populationOld[i, j];
                                num_arc_inds++;

                            }
                            //Whenever the size of the archive exceeds, randomly selected elements are deleted to make space for the newly inserted elements
                            else {
                                rand_arc_ind = (int)(_random.NextDouble() * arc_size);
                                for (int j = 0; j < problem_size; j++) this.archive[rand_arc_ind, j] = populationOld[i, j];
                            }
                        }

                        dif_fitness[num_success_params] = Math.Abs(fitness[i] - children_fitness[i]);

                        fitness[i] = children_fitness[i];
                        for (int j = 0; j < problem_size; j++) populationOld[i, j] = trialPopulation[i, j];

                        //successful parameters are preserved in S_F and S_CR
                        success_sf[num_success_params] = pop_sf[i];
                        success_cr[num_success_params] = pop_cr[i];
                        num_success_params++;
                    }
                }

                if (num_success_params > 0)
                {
                    temp_sum_sf1 = 0;
                    temp_sum_sf2 = 0;
                    temp_sum_cr1 = 0;
                    temp_sum_cr2 = 0;
                    temp_sum = 0;
                    temp_weight = 0;

                    for (int i = 0; i < num_success_params; i++) temp_sum += dif_fitness[i];

                    //weighted lehmer mean
                    for (int i = 0; i < num_success_params; i++)
                    {
                        temp_weight = dif_fitness[i] / temp_sum;

                        temp_sum_sf1 += temp_weight * success_sf[i] * success_sf[i];
                        temp_sum_sf2 += temp_weight * success_sf[i];

                        temp_sum_cr1 += temp_weight * success_cr[i] * success_cr[i];
                        temp_sum_cr2 += temp_weight * success_cr[i];
                    }

                    memory_sf[memory_pos] = temp_sum_sf1 / temp_sum_sf2;

                    if (temp_sum_cr2 == 0 || memory_cr[memory_pos] == -1)
                    {
                        memory_cr[memory_pos] = -1;
                    } else {
                        memory_cr[memory_pos] = temp_sum_cr1 / temp_sum_cr2;
                    }

                    //increment the counter
                    memory_pos++;
                    if (memory_pos >= memory_size) memory_pos = 0;
                }

                //update the best candidate
                for (int i = 0; i < PopulationSizeParameter.Value.Value; i++)
                {
                    selectionVector = new RealVector(getMatrixRow(populationOld, i));
                    var quality = fitness[i];
                    if (quality < bestPopulationValue)
                    {
                        bestPopulationVector = (RealVector)selectionVector.Clone();
                        bestPopulationValue = quality;
                    }
                }

                iterations = iterations + 1;

                //update the results
                ResultsEvaluations = evals;
                ResultsIterations = iterations;
                ResultsBestSolution = bestPopulationVector;
                ResultsBestQuality = bestPopulationValue;

                //update the results in view
                if (iterations % 10 == 0) ResultsQualitiesBest.Values.Add(bestPopulationValue);
                if (bestPopulationValue < Problem.BestKnownQuality.Value + ValueToReachParameter.Value.Value)
                {
                    VTRBestQuality = bestPopulationValue;
                }
            }
        }

        //evaluate the vector
        public double Obj(RealVector x)
        {
            evals = evals + 1;
            if (Problem.Maximization.Value)
                return -Problem.Evaluator.Evaluate(x);

            return Problem.Evaluator.Evaluate(x);
        }

        // Get ith row from the matrix
        public double[] getMatrixRow(double[,] Mat, int i)
        {
            double[] tmp = new double[Mat.GetUpperBound(1) + 1];

            for (int j = 0; j <= Mat.GetUpperBound(1); j++)
            {
                tmp[j] = Mat[i, j];
            }

            return tmp;
        }

        /*
            Return random value from Cauchy distribution with mean "mu" and variance "gamma"
            http://www.sat.t.u-tokyo.ac.jp/~omi/random_variables_generation.html#Cauchy
        */
        private double cauchy_g(double mu, double gamma)
        {
            return mu + gamma * Math.Tan(Math.PI * (_random.NextDouble() - 0.5));
        }

        /*
             Return random value from normal distribution with mean "mu" and variance "gamma"
             http://www.sat.t.u-tokyo.ac.jp/~omi/random_variables_generation.html#Gauss
        */
        private double gauss(double mu, double sigma)
        {
            return mu + sigma * Math.Sqrt(-2.0 * Math.Log(_random.NextDouble())) * Math.Sin(2.0 * Math.PI * _random.NextDouble());
        }

        private double[,] makeNewIndividuals() {
            //problem variables
            var dim = Problem.ProblemSize.Value;
            var lb = Problem.Bounds[0, 0];
            var ub = Problem.Bounds[0, 1];
            var range = ub - lb;
            double[,] population = new double[PopulationSizeParameter.Value.Value, Problem.ProblemSize.Value];

            //create initial population
            //population is a matrix of size PopulationSize*ProblemSize
            for (int i = 0; i < PopulationSizeParameter.Value.Value; i++)
            {
                for (int j = 0; j < Problem.ProblemSize.Value; j++)
                {
                    population[i, j] = _random.NextDouble() * range + lb;
                }
            }
            return population;
        }

        private static void Quicksort(double[] elements, int left, int right, int[] index)
        {
            int i = left, j = right;
            double pivot = elements[(left + right) / 2];
            double tmp_var = 0;
            int tmp_index = 0;

            while (i <= j)
            {
                while (elements[i].CompareTo(pivot) < 0)
                {
                    i++;
                }

                while (elements[j].CompareTo(pivot) > 0)
                {
                    j--;
                }

                if (i <= j)
                {
                    // Swap
                    tmp_var = elements[i];
                    elements[i] = elements[j];
                    elements[j] = tmp_var;

                    tmp_index = index[i];
                    index[i] = index[j];
                    index[j] = tmp_index;

                    i++;
                    j--;
                }
            }

            // Recursive calls
            if (left < j)
            {
                Quicksort(elements, left, j, index);
            }

            if (i < right)
            {
                Quicksort(elements, i, right, index);
            }
        }

        // current to best selection scheme with archive
        // analyze how the archive is implemented
        private double[,] operateCurrentToPBest1BinWithArchive(double[,] pop, double[,]children, int target, int p_best_individual, double scaling_factor, double cross_rate)
        {
            int r1, r2;
            int num_arc_inds = 0;
            var lb = Problem.Bounds[0, 0];
            var ub = Problem.Bounds[0, 1];

            do
            {
                r1 = (int)(_random.NextDouble() * PopulationSizeParameter.Value.Value);
            } while (r1 == target);
            do
            {
                r2 = (int)(_random.NextDouble() * (PopulationSizeParameter.Value.Value + num_arc_inds));
            } while ((r2 == target) || (r2 == r1));

            int random_variable = (int)(_random.NextDouble() * Problem.ProblemSize.Value);

            if (r2 >= PopulationSizeParameter.Value.Value)
            {
                r2 -= PopulationSizeParameter.Value.Value;
                for (int i = 0; i < Problem.ProblemSize.Value; i++)
                {
                    if ((_random.NextDouble() < cross_rate) || (i == random_variable)) children[target, i] = pop[target, i] + scaling_factor * (pop[p_best_individual, i] - pop[target, i]) + scaling_factor * (pop[r1, i] - archive[r2, i]);
                    else children[target, i] = pop[target, i];
                }
            }
            else {
                for (int i = 0; i < Problem.ProblemSize.Value; i++)
                {
                    if ((_random.NextDouble() < cross_rate) || (i == random_variable)) children[target, i] = pop[target, i] + scaling_factor * (pop[p_best_individual, i] - pop[target, i]) + scaling_factor * (pop[r1, i] - pop[r2, i]);
                    else children[target, i] = pop[target, i];
                }
            }

            for (int i = 0; i < Problem.ProblemSize.Value; i++) {
                if (children[target, i] < lb) children[target, i] = (lb + pop[target, i]) / 2.0;
                else if (children[target, i] > ub) children[target, i] = (ub + pop[target, i]) / 2.0;
            }

            return children;
        }
    }
}
