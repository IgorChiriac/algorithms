#region License Information
/* HeuristicLab
 * Copyright (C) 2002-2016 Heuristic and Evolutionary Algorithms Laboratory (HEAL)
 *
 * This file is part of HeuristicLab.
 * Implementation is based on jMetal framework https://github.com/jMetal/jMetal
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

namespace HeuristicLab.Algorithms.DifferentialEvolution
{

    [Item("Differential Evolution (DE)", "A differential evolution algorithm.")]
    [StorableClass]
    [Creatable(CreatableAttribute.Categories.PopulationBasedAlgorithms, Priority = 400)]
    public class DifferentialEvolution : BasicAlgorithm
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

        #region ParameterNames
        private const string MaximumEvaluationsParameterName = "Maximum Evaluations";
        private const string SeedParameterName = "Seed";
        private const string SetSeedRandomlyParameterName = "SetSeedRandomly";
        private const string CrossoverProbabilityParameterName = "CrossoverProbability";
        private const string PopulationSizeParameterName = "PopulationSize";
        private const string ScalingFactorParameterName = "ScalingFactor";
        private const string ValueToReachParameterName = "ValueToReach";
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
        protected DifferentialEvolution(bool deserializing) : base(deserializing) { }

        protected DifferentialEvolution(DifferentialEvolution original, Cloner cloner)
          : base(original, cloner)
        {
        }

        public override IDeepCloneable Clone(Cloner cloner)
        {
            return new DifferentialEvolution(this, cloner);
        }

        public DifferentialEvolution()
        {
            Parameters.Add(new FixedValueParameter<IntValue>(MaximumEvaluationsParameterName, "", new IntValue(Int32.MaxValue)));
            Parameters.Add(new FixedValueParameter<IntValue>(SeedParameterName, "The random seed used to initialize the new pseudo random number generator.", new IntValue(0)));
            Parameters.Add(new FixedValueParameter<BoolValue>(SetSeedRandomlyParameterName, "True if the random seed should be set to a random value, otherwise false.", new BoolValue(true)));
            Parameters.Add(new ValueParameter<IntValue>(PopulationSizeParameterName, "The size of the population of solutions.", new IntValue(100)));
            Parameters.Add(new ValueParameter<DoubleValue>(CrossoverProbabilityParameterName, "The value for crossover rate", new DoubleValue(0.88)));
            Parameters.Add(new ValueParameter<DoubleValue>(ScalingFactorParameterName, "The value for scaling factor", new DoubleValue(0.47)));
            Parameters.Add(new ValueParameter<DoubleValue>(ValueToReachParameterName, "Value to reach (VTR) parameter", new DoubleValue(0.00000001)));
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


            //problem variables
            var dim = Problem.ProblemSize.Value;
            var lb = Problem.Bounds[0, 0];
            var ub = Problem.Bounds[0, 1];
            var range = ub - lb;
            this.evals = 0;

            double[,] populationOld = new double[PopulationSizeParameter.Value.Value, Problem.ProblemSize.Value];
            double[,] mutationPopulation = new double[PopulationSizeParameter.Value.Value, Problem.ProblemSize.Value];
            double[,] trialPopulation = new double[PopulationSizeParameter.Value.Value, Problem.ProblemSize.Value];
            double[] bestPopulation = new double[Problem.ProblemSize.Value];
            double[] bestPopulationIteration = new double[Problem.ProblemSize.Value];

            //create initial population 
            //population is a matrix of size PopulationSize*ProblemSize
            for (int i = 0; i < PopulationSizeParameter.Value.Value; i++)
            {
                for (int j = 0; j < Problem.ProblemSize.Value; j++)
                {
                    populationOld[i, j] = _random.NextDouble() * range + lb;
                }
            }

            //evaluate the best member after the intialiazation
            //the idea is to select first member and after that to check the others members from the population

            int best_index = 0;
            double[] populationRow = new double[Problem.ProblemSize.Value];
            double[] qualityPopulation = new double[PopulationSizeParameter.Value.Value];
            bestPopulation = getMatrixRow(populationOld, best_index);
            RealVector bestPopulationVector = new RealVector(bestPopulation);
            double bestPopulationValue = Obj(bestPopulationVector);
            qualityPopulation[best_index] = bestPopulationValue;
            RealVector selectionVector;
            RealVector trialVector;
            double qtrial;


            for (var i = 1; i < PopulationSizeParameter.Value.Value; i++)
            {
                populationRow = getMatrixRow(populationOld, i);
                trialVector = new RealVector(populationRow);

                qtrial = Obj(trialVector);
                qualityPopulation[i] = qtrial;

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
                && !cancellationToken.IsCancellationRequested
                && (bestPopulationValue - Problem.BestKnownQuality.Value) > ValueToReachParameter.Value.Value)
            {
                //mutation DE/rand/1/bin; classic DE
                for (int i = 0; i < PopulationSizeParameter.Value.Value; i++)
                {
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

                    for (int j = 0; j < getMatrixRow(mutationPopulation, i).Length; j++)
                    {
                        mutationPopulation[i, j] = populationOld[r0, j] +
                            ScalingFactorParameter.Value.Value * (populationOld[r1, j] - populationOld[r2, j]);
                        //check the problem upper and lower bounds
                        if (mutationPopulation[i, j] > ub) mutationPopulation[i, j] = ub;
                        if (mutationPopulation[i, j] < lb) mutationPopulation[i, j] = lb;
                    }
                }

                //uniform crossover
                for (int i = 0; i < PopulationSizeParameter.Value.Value; i++)
                {
                    double rnbr = _random.Next(0, Problem.ProblemSize.Value);
                    for (int j = 0; j < getMatrixRow(mutationPopulation, i).Length; j++)
                    {
                        if (_random.NextDouble() <= CrossoverProbabilityParameter.Value.Value || j == rnbr)
                        {
                            trialPopulation[i, j] = mutationPopulation[i, j];
                        }
                        else
                        {
                            trialPopulation[i, j] = populationOld[i, j];
                        }
                    }
                }

                //One-to-One Survivor Selection
                for (int i = 0; i < PopulationSizeParameter.Value.Value; i++)
                {
                    selectionVector = new RealVector(getMatrixRow(populationOld, i));
                    trialVector = new RealVector(getMatrixRow(trialPopulation, i));

                    var selectionEval = qualityPopulation[i];
                    var trialEval = Obj(trialVector);

                    if (trialEval < selectionEval)
                    {
                        for (int j = 0; j < getMatrixRow(populationOld, i).Length; j++)
                        {
                            populationOld[i, j] = trialPopulation[i, j];
                        }
                        qualityPopulation[i] = trialEval;
                    }
                }

                //update the best candidate
                for (int i = 0; i < PopulationSizeParameter.Value.Value; i++)
                {
                    selectionVector = new RealVector(getMatrixRow(populationOld, i));
                    var quality = qualityPopulation[i];
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
    }
}
