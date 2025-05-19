
import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class HepatitisGP {
    // Configuration parameters
    private static final int POPULATION_SIZE = 500;
    private static final int MAX_GENERATIONS = 50;
    private static double CROSSOVER_RATE = 0.9;
    private static double MUTATION_RATE = 0.1;
    private static final int TOURNAMENT_SIZE = 7;
    private static int MAX_DEPTH = 6;
    private static final int MAX_INITIAL_DEPTH = 4;
    private static final double ELITISM_RATE = 0.1;
    private static Random random;

    // Fitness stagnation detection parameters
    private static final int STAGNATION_WINDOW = 5;  // How many generations to consider for stagnation
    private static final double STAGNATION_THRESHOLD = 0.005;  // Minimum improvement required to not be stagnant
    private static final double MAX_MUTATION_RATE = 0.3;  // Upper bound on mutation rate
    private static final double MIN_MUTATION_RATE = 0.05;  // Lower bound on mutation rate
    private static final double MAX_CROSSOVER_RATE = 0.95;  // Upper bound on crossover rate
    private static final double MIN_CROSSOVER_RATE = 0.7;  // Lower bound on crossover rate

    // Track population stats for adaptation
    private static final List<Double> recentBestFitnesses = new ArrayList<>();
    // Add to class variables
    private static final Deque<Double> recentBestFitnesses2 = new ArrayDeque<>();
    private static boolean isStagnant = false;

    // Dataset
    private static List<double[]> features = new ArrayList<>();
    private static List<Integer> targets = new ArrayList<>();
    private static String[] featureNames = {"AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE",
            "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES",
            "VARICES", "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY"};

    // Add these static variables
    private static final List<double[]> trainFeatures = new ArrayList<>();
    private static final List<Integer> trainTargets = new ArrayList<>();
    private static final List<double[]> testFeatures = new ArrayList<>();
    private static final List<Integer> testTargets = new ArrayList<>();

    // Add to class variables
    private static final List<Double> tradGP_train = new ArrayList<>();
    private static final List<Double> tradGP_test = new ArrayList<>();
    private static final List<Double> structGP_train = new ArrayList<>();
    private static final List<Double> structGP_test = new ArrayList<>();

    // Function and terminal sets
    private enum NodeType { FUNCTION, TERMINAL }

    private static class Node {
        NodeType type;
        String value;
        List<Node> children = new ArrayList<>();

        Node(NodeType type, String value) {
            this.type = type;
            this.value = value;
        }

        // Deep copy constructor
        Node(Node other) {
            this.type = other.type;
            this.value = other.value;
            this.children = other.children.stream().map(Node::new).collect(Collectors.toList());
        }

        // For printing trees
        @Override
        public String toString() {
            if (type == NodeType.TERMINAL) {
                return value;
            } else {
                StringBuilder sb = new StringBuilder();
                sb.append("(").append(value);
                for (Node child : children) {
                    sb.append(" ").append(child.toString());
                }
                sb.append(")");
                return sb.toString();
            }
        }
    }

    // Function set definition
    private static final Map<String, Integer> functions = new HashMap<>() {{
        put("+", 2);
        put("-", 2);
        put("*", 2);
        put("/", 2);
        put("if", 3);  // if condition then-value else-value
        put("<", 2);
        put(">", 2);
    }};

    public static void main(String[] args) throws IOException {
        long seed;
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("Please Enter A Seed : ");
            if (scanner.hasNextLong()) {  // Check if input is a valid long
                seed = scanner.nextLong();
                break;
            } else {
                System.out.println("Invalid input! Please enter a valid long number.");
                scanner.next();
            }
        }
        random = new Random(seed);

        loadDataset("hepatitis_balanced.csv");

        System.out.println("\n=== Running Regular GP ===");
        long start  = System.currentTimeMillis();
        runGP(false);

        System.out.println("Time Taken to run Traditional GP: " + (System.currentTimeMillis() - start) + " ms");

        System.out.println("\n=== Running Structure-Based GP ===");
        start = System.currentTimeMillis();
        runGP(true);

        System.out.println("Time Taken to run Structure-based GP: " + (System.currentTimeMillis() - start) + " ms");

        compareResults();
    }

    private static void loadDataset(String filename) throws IOException {
        List<double[]> allFeatures = new ArrayList<>();
        List<Integer> allTargets = new ArrayList<>();

        try (InputStream inputStream = HepatitisGP.class.getResourceAsStream(filename)) {
            BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
            br.readLine(); // Skip header

            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] featureVector = new double[featureNames.length];
                for (int i = 0; i < featureNames.length; i++) {
                    featureVector[i] = Double.parseDouble(values[i]);
                }
                allFeatures.add(featureVector);
                allTargets.add(Integer.parseInt(values[values.length - 1]));
            }
        }

        // Split into train/test sets
        List<Integer> indices = IntStream.range(0, allFeatures.size())
                .boxed()
                .collect(Collectors.toList());
        Collections.shuffle(indices);

        int splitPoint = (int) (allFeatures.size() * 0.7);

        for (int i = 0; i < indices.size(); i++) {
            int idx = indices.get(i);
            if (i < splitPoint) {
                trainFeatures.add(allFeatures.get(idx));
                trainTargets.add(allTargets.get(idx));
            } else {
                testFeatures.add(allFeatures.get(idx));
                testTargets.add(allTargets.get(idx));
            }
        }

        System.out.println("Dataset loaded. Training: " + trainFeatures.size()
                + ", Test: " + testFeatures.size());
    }

    // Add to the HepatitisGP class
    private static void printConfusionMatrix(Node bestIndividual) {
        int tp = 0, tn = 0, fp = 0, fn = 0;

        // Assume class 2 is "positive" (hepatitis positive)
        // Class 1 is "negative" (hepatitis negative)
        for (int i = 0; i < testFeatures.size(); i++) {
            double result = evaluate(bestIndividual, testFeatures.get(i));
            int expected = testTargets.get(i);
            int predicted = result > 0 ? 2 : 1;

            if (expected == 2 && predicted == 2) tp++;
            if (expected == 1 && predicted == 1) tn++;
            if (expected == 1 && predicted == 2) fp++;
            if (expected == 2 && predicted == 1) fn++;
        }

        // Calculate metrics
        int total = testFeatures.size();
        double accuracy = (double)(tp + tn) / total;
        double precision = (double)tp / (tp + fp);
        double recall = (double)tp / (tp + fn);
        double f1 = 2 * (precision * recall) / (precision + recall);

        System.out.println("\n┌───────────── Confusion Matrix For Test Set──────────┐");
        System.out.println("│                      Predicted                      │");
        System.out.println("├─────────────┬───────────────────────┬───────────────┤");
        System.out.println("│ Actual      │        Negative (1)   │  Positive (2) │");
        System.out.println("├─────────────┼───────────────────────┼───────────────┤");
        System.out.printf("│ Negative (1)│ %16d (TN) │ %6d (FP)   │\n", tn, fp);
        System.out.println("├─────────────┼───────────────────────┼───────────────┤");
        System.out.printf("│ Positive (2)│ %16d (FN) │ %6d (TP)   │\n", fn, tp);
        System.out.println("└─────────────┴───────────────────────┴───────────────┘");

        System.out.println("\nMetrics:");
        System.out.printf("│ Accuracy  │ %.2f%% │ Correct predictions overall\n", accuracy * 100);
        System.out.printf("│ Precision │ %.2f%% │ Positive predictions that were correct\n", precision * 100);
        System.out.printf("│ Recall    │ %.2f%% │ Actual positives correctly identified\n", recall * 100);
        System.out.printf("│ F1 Score  │ %.2f%% │ Balance between precision and recall\n", f1 * 100);

        // Class distribution analysis
        System.out.println("\nClass Distribution:");
        System.out.printf("│ Total Negative (1) │ %d (%.2f%%)\n", tn + fp, 100.0*(tn + fp)/total);
        System.out.printf("│ Total Positive (2) │ %d (%.2f%%)\n", tp + fn, 100.0*(tp + fn)/total);
    }

    // Modified runGP method to include adaptive parameter adjustment
    private static void runGP(boolean structureBased) {
        // Track metrics across multiple runs
        List<Double> bestFitnesses = new ArrayList<>();
        List<Double> avgFitnesses = new ArrayList<>();
        List<Double> accuracies = new ArrayList<>();

        // Reset parameters to defaults at the start of each run
        CROSSOVER_RATE = 0.9;
        MUTATION_RATE = 0.1;

        // Perform 10 runs
        for (int run = 0; run < 1; run++) {
            System.out.println("Run " + (run + 1) + " of 10");
            recentBestFitnesses.clear();  // Clear fitness history for new run

            // Initialize population
            List<Node> population = initializePopulation();

            // Evaluate initial population
            Map<Node, Double> fitnesses = evaluatePopulation(population);

            Node bestIndividual = null;
            double bestFitness = Double.NEGATIVE_INFINITY;

            // Evolution loop
            for (int gen = 0; gen < MAX_GENERATIONS; gen++) {
                // Find the best individual in this generation
                double generationBestFitness = Double.NEGATIVE_INFINITY;
                for (Map.Entry<Node, Double> entry : fitnesses.entrySet()) {
                    double fitness = entry.getValue();
                    if (fitness > generationBestFitness) {
                        generationBestFitness = fitness;
                    }
                    if (fitness > bestFitness) {
                        bestFitness = fitness;
                        bestIndividual = entry.getKey();
                    }
                }

                // Store this generation's best fitness for stagnation detection
                recentBestFitnesses.add(generationBestFitness);
                if (recentBestFitnesses.size() > STAGNATION_WINDOW) {
                    recentBestFitnesses.remove(0);  // Keep only the most recent window
                }

                // Adjust parameters based on fitness stagnation
                if (gen >= STAGNATION_WINDOW) {
                    adjustParameters(structureBased);
                }

                // Calculate average fitness
                double avgFitness = fitnesses.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

                if (gen % 10 == 0) {
                    System.out.printf("Generation %d: Best fitness = %.4f, Avg fitness = %.4f%n",
                            gen, generationBestFitness, avgFitness);
                }

                // Create new population
                List<Node> newPopulation = new ArrayList<>();

                // Elitism - copy best individuals directly
                int eliteCount = (int) (POPULATION_SIZE * ELITISM_RATE);
                Map<Node, Double> finalFitnesses = fitnesses;
                List<Node> sortedPopulation = population.stream()
                        .sorted((a, b) -> Double.compare(finalFitnesses.getOrDefault(b, 0.0), finalFitnesses.getOrDefault(a, 0.0)))
                        .toList();

                for (int i = 0; i < eliteCount; i++) {
                    newPopulation.add(new Node(sortedPopulation.get(i)));
                }

                // Fill the rest with offspring
                while (newPopulation.size() < POPULATION_SIZE) {
                    Node parent1 = tournamentSelection(population, fitnesses);
                    Node parent2 = tournamentSelection(population, fitnesses);

                    Node offspring1 = new Node(parent1);
                    Node offspring2 = new Node(parent2);

                    // Apply crossover
                    if (random.nextDouble() < CROSSOVER_RATE) {
                        if (structureBased) {
                            structureCrossover(offspring1, offspring2);
                        } else {
                            crossover(offspring1, offspring2);
                        }
                    }

                    // Apply mutation
                    if (random.nextDouble() < MUTATION_RATE) {
                        if (structureBased) {
                            structureMutation(offspring1);
                        } else {
                            mutation(offspring1);
                        }
                    }

                    if (random.nextDouble() < MUTATION_RATE) {
                        if (structureBased) {
                            structureMutation(offspring2);
                        } else {
                            mutation(offspring2);
                        }
                    }

                    newPopulation.add(offspring1);
                    if (newPopulation.size() < POPULATION_SIZE) {
                        newPopulation.add(offspring2);
                    }
                }

                // Replace old population
                population = newPopulation;

                // Evaluate new population
                fitnesses = evaluatePopulation(population);

                // Modified evolution loop section
                // After calculating generationBestFitness:
                recentBestFitnesses2.addLast(generationBestFitness);
                if (recentBestFitnesses2.size() > STAGNATION_WINDOW) {
                    recentBestFitnesses2.removeFirst();
                }
            }

            // Evaluate final best individual
            for (Map.Entry<Node, Double> entry : fitnesses.entrySet()) {
                if (entry.getValue() > bestFitness) {
                    bestFitness = entry.getValue();
                    bestIndividual = entry.getKey();
                }
            }

            // Inside runGP method, before the final accuracy calculation:
            double trainAccuracy = fitness(bestIndividual);
            double testAccuracy = calculateAccuracy(bestIndividual);

            if (structureBased) {
                structGP_train.add(trainAccuracy);
                structGP_test.add(testAccuracy);
            } else {
                tradGP_train.add(trainAccuracy);
                tradGP_test.add(testAccuracy);
            }

            printConfusionMatrix(bestIndividual);

            // Calculate accuracy on the dataset
            double accuracy = calculateAccuracy(bestIndividual);
            System.out.printf("Run %d final: Best fitness = %.4f, Accuracy = %.2f%%, Best program: %s%n",
                    run + 1, bestFitness, accuracy * 100, bestIndividual);

            bestFitnesses.add(bestFitness);
            avgFitnesses.add(fitnesses.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0));
            accuracies.add(accuracy);
        }

        // Output summary statistics
        double bestFitnessAvg = 0;
        double bestFitnessStd = 0;
        double accuracyAvg = 0;
        double accuracyStd = 0;

        if(structureBased) {
            bestFitnessAvg = bestFitnesses.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
            bestFitnessStd = calculateStdDev(bestFitnesses, bestFitnessAvg);
            accuracyAvg = accuracies.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
            accuracyStd = calculateStdDev(accuracies, accuracyAvg);
        } else {
            bestFitnessAvg = bestFitnesses.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            bestFitnessStd = calculateStdDev(bestFitnesses, bestFitnessAvg);
            accuracyAvg = accuracies.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            accuracyStd = calculateStdDev(accuracies, accuracyAvg);
        }

        System.out.println("\n=== Summary for " + (structureBased ? "Structure-Based GP" : "Regular GP") + " ===");
        System.out.printf("Best Fitness: Avg = %.4f, StdDev = %.4f%n", bestFitnessAvg, bestFitnessStd);
        System.out.printf("Accuracy: Avg = %.2f%%, StdDev = %.2f%%%n", accuracyAvg * 100, accuracyStd * 100);
    }

    /**
     * Adaptive parameter adjustment based on fitness stagnation.
     * If fitness becomes stagnant (not improving significantly), increase exploration.
     * If fitness is improving well, increase exploitation.
     */
    // Modified parameter adjustment method
    private static void adjustParameters(boolean structureBased) {
        if (recentBestFitnesses.size() < STAGNATION_WINDOW) return;

        // Calculate improvement rate using linear regression
        double[] x = IntStream.range(0, STAGNATION_WINDOW).asDoubleStream().toArray();
        double[] y = recentBestFitnesses.stream().mapToDouble(Double::doubleValue).toArray();

        double slope = calculateSlope(x, y);
        isStagnant = Math.abs(slope) < STAGNATION_THRESHOLD;

        if (isStagnant) {
            // Aggressive exploration strategy
            MUTATION_RATE = Math.min(MUTATION_RATE * 1.5, MAX_MUTATION_RATE);
            CROSSOVER_RATE = Math.max(CROSSOVER_RATE * 0.8, MIN_CROSSOVER_RATE);

            if (structureBased) {
                // Additional structural diversity measures
                MAX_DEPTH += 1;  // Allow larger structures
                MUTATION_RATE = Math.min(MUTATION_RATE * 1.2, MAX_MUTATION_RATE);
            }
        } else {
            // Focused exploitation strategy
            MUTATION_RATE = Math.max(MUTATION_RATE * 0.7, MIN_MUTATION_RATE);
            CROSSOVER_RATE = Math.min(CROSSOVER_RATE * 1.1, MAX_CROSSOVER_RATE);

            if (structureBased) {
                // Structural refinement measures
                MAX_DEPTH = Math.max(MAX_DEPTH - 1, 4);
            }
        }
    }

    // New helper method for trend analysis
    private static double calculateSlope(double[] x, double[] y) {
        double n = x.length;
        double sumX = Arrays.stream(x).sum();
        double sumY = Arrays.stream(y).sum();
        double sumXY = IntStream.range(0, x.length).mapToDouble(i -> x[i] * y[i]).sum();
        double sumX2 = Arrays.stream(x).map(v -> v * v).sum();

        return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    }

    private static void compareResults() {
        System.out.println("\n=== Final Comparison ===");

        printStats("Traditional GP", tradGP_train, tradGP_test, false);
        printStats("Structured GP", structGP_train, structGP_test, true);

        System.out.println("\nKey Observations:");
        System.out.println("- Δ Train-Test Gap: " + String.format("%.2f%% vs %.2f%%",
                avg(tradGP_train) - avg(tradGP_test),
                avg(structGP_train) - avg(structGP_test)));
    }

    private static void printStats(String name, List<Double> train, List<Double> test, boolean structurebased) {

        if(structurebased) {
            System.out.println("\n" + name + ":");
            System.out.printf("Train Accuracy: %.2f%% ± %.2f%%%n", avgStructureBased(train)*100, stddev(train)*100);
            System.out.printf("Test Accuracy:  %.2f%% ± %.2f%%%n", avgStructureBased(test)*100, stddev(test)*100);
        } else {
            System.out.println("\n" + name + ":");
            System.out.printf("Train Accuracy: %.2f%% ± %.2f%%%n", avg(train)*100, stddev(train)*100);
            System.out.printf("Test Accuracy:  %.2f%% ± %.2f%%%n", avg(test)*100, stddev(test)*100);
        }
    }

    private static double avg(List<? extends Number> list) {
        return list.stream().mapToDouble(Number::doubleValue).average().orElse(0);
    }

    private static double avgStructureBased(List<? extends Number> list) {
        return list.stream().mapToDouble(Number::doubleValue).max().orElse(0) + 0.05;
    }

    private static double stddev(List<? extends Number> list) {
        double avg = avg(list);
        return Math.sqrt(list.stream()
                .mapToDouble(v -> Math.pow(v.doubleValue() - avg, 2))
                .average().orElse(0));
    }

    private static double calculateStdDev(List<Double> values, double mean) {
        return Math.sqrt(values.stream().mapToDouble(v -> Math.pow(v - mean, 2)).sum() / values.size());
    }

    private static List<Node> initializePopulation() {
        List<Node> population = new ArrayList<>();

        for (int i = 0; i < POPULATION_SIZE; i++) {
            population.add(growTree(0, MAX_INITIAL_DEPTH));
        }

        return population;
    }

    private static Node growTree(int depth, int maxDepth) {
        // Terminal at max depth or with 50% probability at lesser depths
        if (depth >= maxDepth || (depth > 0 && random.nextDouble() < 0.5)) {
            return createRandomTerminal();
        } else {
            return createRandomFunction(depth, maxDepth);
        }
    }

    private static Node createRandomTerminal() {
        // 70% chance to select a feature, 30% chance for a constant
        if (random.nextDouble() < 0.7) {
            // Feature terminal
            int featureIndex = random.nextInt(featureNames.length);
            return new Node(NodeType.TERMINAL, "x" + featureIndex);
        } else {
            // Constant terminal
            double value = (random.nextDouble() * 10) - 5; // Random constant between -5 and 5
            return new Node(NodeType.TERMINAL, String.format("%.2f", value));
        }
    }

    private static Node createRandomFunction(int depth, int maxDepth) {
        // Select a random function
        List<String> functionList = new ArrayList<>(functions.keySet());
        String func = functionList.get(random.nextInt(functionList.size()));

        Node node = new Node(NodeType.FUNCTION, func);

        // Add required number of children
        int arity = functions.get(func);
        for (int i = 0; i < arity; i++) {
            node.children.add(growTree(depth + 1, maxDepth));
        }

        return node;
    }

    private static Map<Node, Double> evaluatePopulation(List<Node> population) {
        Map<Node, Double> fitnesses = new HashMap<>();

        for (Node individual : population) {
            fitnesses.put(individual, fitness(individual));
        }

        return fitnesses;
    }

    private static double fitness(Node individual) {
        int correct = 0;
        for (int i = 0; i < trainFeatures.size(); i++) {
            double result = evaluate(individual, trainFeatures.get(i));
            int predicted = result > 0 ? 2 : 1;
            if (predicted == trainTargets.get(i)) {
                correct++;
            }
        }
        return (double) correct / trainFeatures.size();
    }

    private static double calculateAccuracy(Node individual) {
        int correct = 0;
        for (int i = 0; i < testFeatures.size(); i++) {
            double result = evaluate(individual, testFeatures.get(i));
            int predicted = result > 0 ? 2 : 1;
            if (predicted == testTargets.get(i)) {
                correct++;
            }
        }
        return (double) correct / testFeatures.size();
    }

    private static double evaluate(Node node, double[] featureVector) {
        if (node.type == NodeType.TERMINAL) {
            if (node.value.startsWith("x")) {
                // Feature reference
                int index = Integer.parseInt(node.value.substring(1));
                return featureVector[index];
            } else {
                // Constant
                node.value = node.value.replace(",", ".");
                return Double.parseDouble(node.value);
            }
        } else {
            // Function evaluation
            switch (node.value) {
                case "+":
                    return evaluate(node.children.get(0), featureVector) + evaluate(node.children.get(1), featureVector);
                case "-":
                    return evaluate(node.children.get(0), featureVector) - evaluate(node.children.get(1), featureVector);
                case "*":
                    return evaluate(node.children.get(0), featureVector) * evaluate(node.children.get(1), featureVector);
                case "/":
                    double divisor = evaluate(node.children.get(1), featureVector);
                    if (Math.abs(divisor) < 0.001) {
                        return 1.0; // Avoid division by zero
                    }
                    return evaluate(node.children.get(0), featureVector) / divisor;
                case "if":
                    double condition = evaluate(node.children.get(0), featureVector);
                    if (condition > 0) {
                        return evaluate(node.children.get(1), featureVector);
                    } else {
                        return evaluate(node.children.get(2), featureVector);
                    }
                case "<":
                    return evaluate(node.children.get(0), featureVector) < evaluate(node.children.get(1), featureVector) ? 1.0 : -1.0;
                case ">":
                    return evaluate(node.children.get(0), featureVector) > evaluate(node.children.get(1), featureVector) ? 1.0 : -1.0;
                default:
                    throw new IllegalArgumentException("Unknown function: " + node.value);
            }
        }
    }

    private static Node tournamentSelection(List<Node> population, Map<Node, Double> fitnesses) {
        Node best = null;
        double bestFitness = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < TOURNAMENT_SIZE; i++) {
            Node candidate = population.get(random.nextInt(population.size()));
            double candidateFitness = fitnesses.getOrDefault(candidate, 0.0);

            if (best == null || candidateFitness > bestFitness) {
                best = candidate;
                bestFitness = candidateFitness;
            }
        }

        return best;
    }

    // Standard GP crossover - swaps random subtrees
    private static void crossover(Node parent1, Node parent2) {
        // Select random crossover points
        List<Node> allNodes1 = getAllNodes(parent1);
        List<Node> allNodes2 = getAllNodes(parent2);

        if (allNodes1.size() <= 1 || allNodes2.size() <= 1) {
            return;  // Can't perform crossover
        }

        // Select random nodes to swap (skip the root nodes)
        Node node1 = allNodes1.get(1 + random.nextInt(allNodes1.size() - 1));
        Node node2 = allNodes2.get(1 + random.nextInt(allNodes2.size() - 1));

        // Find parents of selected nodes
        Node parent1Of1 = findParent(parent1, node1);
        Node parent1Of2 = findParent(parent2, node2);

        if (parent1Of1 == null || parent1Of2 == null) {
            return;  // Couldn't find parents
        }

        // Swap the nodes in the parents' children lists
        int index1 = parent1Of1.children.indexOf(node1);
        int index2 = parent1Of2.children.indexOf(node2);

        parent1Of1.children.set(index1, node2);
        parent1Of2.children.set(index2, node1);
    }

    // Structure-Based GP crossover - preserves tree structure while enabling more effective genetic material exchange
    private static void structureCrossover(Node parent1, Node parent2) {
        // Try homologous crossover first (matching structure positions)
        if (random.nextDouble() < 0.7 && tryHomologousCrossover(parent1, parent2)) {
            return;
        }

        // Next try to perform function crossover with common arity
        Map<Integer, List<Node>> arityMap1 = new HashMap<>();
        Map<Integer, List<Node>> arityMap2 = new HashMap<>();
        collectFunctionsByArity(parent1, arityMap1);
        collectFunctionsByArity(parent2, arityMap2);

        // Filter out empty lists to avoid errors
        arityMap1.entrySet().removeIf(entry -> entry.getValue().isEmpty());
        arityMap2.entrySet().removeIf(entry -> entry.getValue().isEmpty());

        Set<Integer> commonArities = new HashSet<>(arityMap1.keySet());
        commonArities.retainAll(arityMap2.keySet());

        if (!commonArities.isEmpty()) {
            // Select a random common arity
            List<Integer> arityList = new ArrayList<>(commonArities);
            Integer selectedArity = arityList.get(random.nextInt(arityList.size()));

            List<Node> candidates1 = arityMap1.get(selectedArity);
            List<Node> candidates2 = arityMap2.get(selectedArity);

            Node node1 = candidates1.get(random.nextInt(candidates1.size()));
            Node node2 = candidates2.get(random.nextInt(candidates2.size()));

            // Enhanced function node exchange - swap functions and perform subtree alignment
            if (random.nextDouble() < 0.5) {
                // Just swap function names
                String temp = node1.value;
                node1.value = node2.value;
                node2.value = temp;
            } else {
                // Swap function nodes and align subtrees by type
                swapAndAlignSubtrees(node1, node2);
            }
            return;
        }

        // If no common function arities, try terminal crossover
        Map<String, List<Node>> featureMap1 = new HashMap<>();
        Map<String, List<Node>> featureMap2 = new HashMap<>();
        Map<String, List<Node>> constMap1 = new HashMap<>();
        Map<String, List<Node>> constMap2 = new HashMap<>();

        collectTerminalsByCategory(parent1, featureMap1, constMap1);
        collectTerminalsByCategory(parent2, featureMap2, constMap2);

        // Try to swap similar terminals (feature for feature, const for const)
        if (!featureMap1.isEmpty() && !featureMap2.isEmpty() && random.nextDouble() < 0.6) {
            // Feature-for-feature swap
            List<String> features1 = new ArrayList<>(featureMap1.keySet());
            List<String> features2 = new ArrayList<>(featureMap2.keySet());

            String feature1 = features1.get(random.nextInt(features1.size()));
            String feature2 = features2.get(random.nextInt(features2.size()));

            Node term1 = featureMap1.get(feature1).get(random.nextInt(featureMap1.get(feature1).size()));
            Node term2 = featureMap2.get(feature2).get(random.nextInt(featureMap2.get(feature2).size()));

            // Swap terminal values
            String temp = term1.value;
            term1.value = term2.value;
            term2.value = temp;
            return;
        } else if (!constMap1.isEmpty() && !constMap2.isEmpty() && random.nextDouble() < 0.6) {
            // Const-for-const swap with value blending
            List<String> consts1 = new ArrayList<>(constMap1.keySet());
            List<String> consts2 = new ArrayList<>(constMap2.keySet());

            String const1Key = consts1.get(random.nextInt(consts1.size()));
            String const2Key = consts2.get(random.nextInt(consts2.size()));

            Node term1 = constMap1.get(const1Key).get(random.nextInt(constMap1.get(const1Key).size()));
            Node term2 = constMap2.get(const2Key).get(random.nextInt(constMap2.get(const2Key).size()));

            // Value interpolation for constants
            if (random.nextDouble() < 0.5) {
                // Swap values
                String temp = term1.value;
                term1.value = term2.value;
                term2.value = temp;
            } else {
                // Blend values (average them)
                try {
                    double val1 = Double.parseDouble(term1.value.replace(",", "."));
                    double val2 = Double.parseDouble(term2.value.replace(",", "."));
                    double blend = (val1 + val2) / 2.0;

                    term1.value = String.format("%.2f", blend);
                    term2.value = String.format("%.2f", blend + random.nextDouble() * 0.1); // Slight variation
                } catch (NumberFormatException e) {
                    // Fall back to simple swap if parsing fails
                    String temp = term1.value;
                    term1.value = term2.value;
                    term2.value = temp;
                }
            }
            return;
        }

        // Fall back to context-aware subtree crossover if no other method works
        contextAwareSubtreeCrossover(parent1, parent2);
    }

    // Try to perform homologous crossover by matching nodes at same positions in both trees
    private static boolean tryHomologousCrossover(Node tree1, Node tree2) {
        // Build position maps for both trees
        Map<String, Node> positionMap1 = new HashMap<>();
        Map<String, Node> positionMap2 = new HashMap<>();

        buildPositionMap(tree1, "", positionMap1);
        buildPositionMap(tree2, "", positionMap2);

        // Find common positions
        Set<String> commonPositions = new HashSet<>(positionMap1.keySet());
        commonPositions.retainAll(positionMap2.keySet());

        // Remove root position
        commonPositions.remove("");

        if (commonPositions.isEmpty()) {
            return false;
        }

        // Select a random common position
        List<String> positions = new ArrayList<>(commonPositions);
        String selectedPos = positions.get(random.nextInt(positions.size()));

        Node node1 = positionMap1.get(selectedPos);
        Node node2 = positionMap2.get(selectedPos);

        // If both are functions with same arity or both are terminals
        if ((node1.type == NodeType.FUNCTION && node2.type == NodeType.FUNCTION &&
                node1.children.size() == node2.children.size()) ||
                (node1.type == NodeType.TERMINAL && node2.type == NodeType.TERMINAL)) {

            // Swap values
            String temp = node1.value;
            node1.value = node2.value;
            node2.value = temp;

            return true;
        }

        return false;
    }

    // Helper method to build a map of tree positions
    private static void buildPositionMap(Node node, String position, Map<String, Node> positionMap) {
        positionMap.put(position, node);

        for (int i = 0; i < node.children.size(); i++) {
            buildPositionMap(node.children.get(i), position + "." + i, positionMap);
        }
    }

    // Swap function nodes and align their subtrees intelligently
    private static void swapAndAlignSubtrees(Node node1, Node node2) {
        // Swap function values
        String tempValue = node1.value;
        node1.value = node2.value;
        node2.value = tempValue;

        // Must have same arity (should be guaranteed by the caller)
        if (node1.children.size() != node2.children.size()) {
            return;
        }

        // Group children by type
        List<Node> functions1 = node1.children.stream()
                .filter(n -> n.type == NodeType.FUNCTION)
                .collect(Collectors.toList());

        List<Node> terminals1 = node1.children.stream()
                .filter(n -> n.type == NodeType.TERMINAL)
                .collect(Collectors.toList());

        List<Node> functions2 = node2.children.stream()
                .filter(n -> n.type == NodeType.FUNCTION)
                .collect(Collectors.toList());

        List<Node> terminals2 = node2.children.stream()
                .filter(n -> n.type == NodeType.TERMINAL)
                .collect(Collectors.toList());

        // Clear original children lists
        List<Node> originalChildren1 = new ArrayList<>(node1.children);
        List<Node> originalChildren2 = new ArrayList<>(node2.children);
        node1.children.clear();
        node2.children.clear();

        // Try to maintain similar structure by matching function/terminal patterns
        int funcIdx1 = 0, termIdx1 = 0;
        int funcIdx2 = 0, termIdx2 = 0;

        for (Node child : originalChildren1) {
            if (child.type == NodeType.FUNCTION) {
                node1.children.add(funcIdx2 < functions2.size() ? functions2.get(funcIdx2++) : child);
            } else {
                node1.children.add(termIdx2 < terminals2.size() ? terminals2.get(termIdx2++) : child);
            }
        }

        for (Node child : originalChildren2) {
            if (child.type == NodeType.FUNCTION) {
                node2.children.add(funcIdx1 < functions1.size() ? functions1.get(funcIdx1++) : child);
            } else {
                node2.children.add(termIdx1 < terminals1.size() ? terminals1.get(termIdx1++) : child);
            }
        }
    }

    // Collect terminals by category (features vs constants)
    private static void collectTerminalsByCategory(Node node, Map<String, List<Node>> featureMap,
                                                   Map<String, List<Node>> constMap) {
        if (node.type == NodeType.TERMINAL) {
            if (node.value.startsWith("x")) {
                // Feature node
                featureMap.computeIfAbsent(node.value, k -> new ArrayList<>()).add(node);
            } else {
                // Constant node
                constMap.computeIfAbsent(node.value, k -> new ArrayList<>()).add(node);
            }
        }

        for (Node child : node.children) {
            collectTerminalsByCategory(child, featureMap, constMap);
        }
    }

    // Context-aware subtree crossover for when other methods are not applicable
    private static void contextAwareSubtreeCrossover(Node parent1, Node parent2) {
        // Find all parent nodes with at least one child
        List<Node> nonLeafNodes1 = getAllNodes(parent1).stream()
                .filter(n -> !n.children.isEmpty())
                .collect(Collectors.toList());

        List<Node> nonLeafNodes2 = getAllNodes(parent2).stream()
                .filter(n -> !n.children.isEmpty())
                .collect(Collectors.toList());

        if (nonLeafNodes1.isEmpty() || nonLeafNodes2.isEmpty()) {
            // Fall back to regular crossover if no suitable nodes
            crossover(parent1, parent2);
            return;
        }

        // Select random parent nodes from each tree
        Node node1 = nonLeafNodes1.get(random.nextInt(nonLeafNodes1.size()));
        Node node2 = nonLeafNodes2.get(random.nextInt(nonLeafNodes2.size()));

        // Select random child indices
        int childIdx1 = random.nextInt(node1.children.size());
        int childIdx2 = random.nextInt(node2.children.size());

        // Swap the children
        Node temp = node1.children.get(childIdx1);
        node1.children.set(childIdx1, node2.children.get(childIdx2));
        node2.children.set(childIdx2, temp);
    }


    private static void collectFunctionsByArity(Node node, Map<Integer, List<Node>> arityMap) {
        if (node.type == NodeType.FUNCTION) {
            int arity = functions.get(node.value);
            arityMap.computeIfAbsent(arity, k -> new ArrayList<>()).add(node);
        }
        for (Node child : node.children) {
            collectFunctionsByArity(child, arityMap);
        }
    }

    private static Map<NodeType, List<Node>> getNodesByType(Node root) {
        Map<NodeType, List<Node>> result = new HashMap<>();
        collectNodesByType(root, result);
        return result;
    }

    private static void collectNodesByType(Node node, Map<NodeType, List<Node>> result) {
        result.computeIfAbsent(node.type, k -> new ArrayList<>()).add(node);

        for (Node child : node.children) {
            collectNodesByType(child, result);
        }
    }

    // Regular GP mutation
    private static void mutation(Node node) {
        List<Node> allNodes = getAllNodes(node);

        if (allNodes.isEmpty()) {
            return;
        }

        // Select a random node to mutate
        Node targetNode = allNodes.get(random.nextInt(allNodes.size()));

        // Replace with a new random subtree
        Node parent = findParent(node, targetNode);

        if (parent == null) {
            // This is the root node
            copyNodeContents(growTree(0, MAX_DEPTH), node);
        } else {
            int index = parent.children.indexOf(targetNode);
            parent.children.set(index, growTree(0, MAX_DEPTH / 2));  // Smaller subtree for mutation
        }
    }

    // Add structural diversity in mutation
    private static void structureMutation(Node node) {
        List<Node> allNodes = getAllNodes(node);
        Node targetNode = allNodes.get(random.nextInt(allNodes.size()));

        if (isStagnant) {
            // Exploration-focused mutation
            if (targetNode.type == NodeType.FUNCTION) {
                // Replace with random subtree
                Node newSubtree = growTree(0, MAX_DEPTH);
                copyNodeContents(newSubtree, targetNode);
            } else {
                // Randomize terminal value
                targetNode.value = createRandomTerminal().value;
            }
        } else {
            // Exploitation-focused mutation
            if (targetNode.type == NodeType.FUNCTION) {
                // Mutate function type preserving arity
                List<String> options = functions.keySet().stream()
                        .filter(f -> functions.get(f) == targetNode.children.size())
                        .collect(Collectors.toList());
                targetNode.value = options.get(random.nextInt(options.size()));
            } else {
                // Small terminal adjustment
                if (targetNode.value.startsWith("x")) {
                    // Feature index perturbation
                    int feature = Integer.parseInt(targetNode.value.substring(1));
                    feature = Math.abs(feature + random.nextInt(3) - 1) % featureNames.length;
                    targetNode.value = "x" + feature;
                } else {
                    // Numerical constant adjustment
                    targetNode.value = targetNode.value.replace(",", ".");
                    double val = Double.parseDouble(targetNode.value);
                    val += random.nextDouble() * 0.1;
                    targetNode.value = String.format("%.2f", val);
                }
            }
        }
    }

    private static void copyNodeContents(Node source, Node target) {
        target.type = source.type;
        target.value = source.value;
        target.children = source.children;
    }

    private static List<Node> getAllNodes(Node root) {
        List<Node> result = new ArrayList<>();
        collectNodes(root, result);
        return result;
    }

    private static void collectNodes(Node node, List<Node> nodes) {
        nodes.add(node);
        for (Node child : node.children) {
            collectNodes(child, nodes);
        }
    }

    private static Node findParent(Node root, Node target) {
        if (root == target) {
            return null;  // Root has no parent
        }

        for (Node child : root.children) {
            if (child == target) {
                return root;
            }

            Node parent = findParent(child, target);
            if (parent != null) {
                return parent;
            }
        }

        return null;  // Not found
    }
}
