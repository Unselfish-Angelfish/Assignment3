
import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class HepatitisGP {
    // Configuration parameters
    private static final int POPULATION_SIZE = 500;
    private static final int MAX_GENERATIONS = 50;
    private static final double CROSSOVER_RATE = 0.9;
    private static final double MUTATION_RATE = 0.1;
    private static final int TOURNAMENT_SIZE = 7;
    private static final int MAX_DEPTH = 6;
    private static final int MAX_INITIAL_DEPTH = 4;
    private static final double ELITISM_RATE = 0.1;

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

    // Terminal set will include feature indices and ephemeral random constants

    // Random number generator
    private static final Random random = new Random();

    public static void main(String[] args) throws IOException {
        loadDataset("hepatitis.csv");

        System.out.println("\n=== Running Regular GP ===");
        runGP(false);

        System.out.println("\n=== Running Structure-Based GP ===");
        runGP(true);

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

    private static void runGP(boolean structureBased) {
        // Track metrics across multiple runs
        List<Double> bestFitnesses = new ArrayList<>();
        List<Double> avgFitnesses = new ArrayList<>();
        List<Double> accuracies = new ArrayList<>();

        // Perform 10 runs
        for (int run = 0; run < 1; run++) {
            System.out.println("Run " + (run + 1) + " of 10");

            // Initialize population
            List<Node> population = initializePopulation();

            // Evaluate initial population
            Map<Node, Double> fitnesses = evaluatePopulation(population);

            Node bestIndividual = null;
            double bestFitness = Double.NEGATIVE_INFINITY;

            // Evolution loop
            for (int gen = 0; gen < MAX_GENERATIONS; gen++) {
                // Find the best individual in this generation
                for (Map.Entry<Node, Double> entry : fitnesses.entrySet()) {
                    if (entry.getValue() > bestFitness) {
                        bestFitness = entry.getValue();
                        bestIndividual = entry.getKey();
                    }
                }

                // Calculate average fitness
                double avgFitness = fitnesses.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

                if (gen % 10 == 0) {
                    System.out.printf("Generation %d: Best fitness = %.4f, Avg fitness = %.4f%n",
                            gen, bestFitness, avgFitness);
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
        double bestFitnessAvg = bestFitnesses.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double bestFitnessStd = calculateStdDev(bestFitnesses, bestFitnessAvg);
        double accuracyAvg = accuracies.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double accuracyStd = calculateStdDev(accuracies, accuracyAvg);

        System.out.println("\n=== Summary for " + (structureBased ? "Structure-Based GP" : "Regular GP") + " ===");
        System.out.printf("Best Fitness: Avg = %.4f, StdDev = %.4f%n", bestFitnessAvg, bestFitnessStd);
        System.out.printf("Accuracy: Avg = %.2f%%, StdDev = %.2f%%%n", accuracyAvg * 100, accuracyStd * 100);
    }

    private static void compareResults() {
        System.out.println("\n=== Final Comparison ===");

        printStats("Traditional GP", tradGP_train, tradGP_test);
        printStats("Structured GP", structGP_train, structGP_test);

        System.out.println("\nKey Observations:");
        System.out.println("- Δ Train-Test Gap: " + String.format("%.2f%% vs %.2f%%",
                avg(tradGP_train) - avg(tradGP_test),
                avg(structGP_train) - avg(structGP_test)));
    }

    private static void printStats(String name, List<Double> train, List<Double> test) {
        System.out.println("\n" + name + ":");
        System.out.printf("Train Accuracy: %.2f%% ± %.2f%%%n", avg(train)*100, stddev(train)*100);
        System.out.printf("Test Accuracy:  %.2f%% ± %.2f%%%n", avg(test)*100, stddev(test)*100);
    }

    private static double avg(List<? extends Number> list) {
        return list.stream().mapToDouble(Number::doubleValue).average().orElse(0);
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

    // Structure-Based GP crossover - preserves tree structure
    private static void structureCrossover(Node parent1, Node parent2) {
        // First try to perform function crossover with common arity
        Map<Integer, List<Node>> arityMap1 = new HashMap<>();
        Map<Integer, List<Node>> arityMap2 = new HashMap<>();
        collectFunctionsByArity(parent1, arityMap1);
        collectFunctionsByArity(parent2, arityMap2);

        Set<Integer> commonArities = new HashSet<>(arityMap1.keySet());
        commonArities.retainAll(arityMap2.keySet());

        if (!commonArities.isEmpty()) {
            Integer selectedArity = commonArities.stream()
                    .skip(random.nextInt(commonArities.size()))
                    .findFirst()
                    .orElse(null);

            List<Node> candidates1 = arityMap1.get(selectedArity);
            List<Node> candidates2 = arityMap2.get(selectedArity);

            Node node1 = candidates1.get(random.nextInt(candidates1.size()));
            Node node2 = candidates2.get(random.nextInt(candidates2.size()));

            // Swap function names
            String temp = node1.value;
            node1.value = node2.value;
            node2.value = temp;
            return;
        }

        // If no common function arities, try terminal crossover
        Map<NodeType, List<Node>> typeNodes1 = getNodesByType(parent1);
        Map<NodeType, List<Node>> typeNodes2 = getNodesByType(parent2);

        List<Node> terminals1 = typeNodes1.getOrDefault(NodeType.TERMINAL, Collections.emptyList());
        List<Node> terminals2 = typeNodes2.getOrDefault(NodeType.TERMINAL, Collections.emptyList());

        if (!terminals1.isEmpty() && !terminals2.isEmpty()) {
            Node term1 = terminals1.get(random.nextInt(terminals1.size()));
            Node term2 = terminals2.get(random.nextInt(terminals2.size()));
            // Swap terminal values
            String temp = term1.value;
            term1.value = term2.value;
            term2.value = temp;
            return;
        }

        // Fall back to regular crossover if no structure-based crossover possible
        crossover(parent1, parent2);
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

    // Structure-based GP mutation
    private static void structureMutation(Node node) {
        List<Node> allNodes = getAllNodes(node);

        if (allNodes.isEmpty()) {
            return;
        }

        // Select a random node to mutate
        Node targetNode = allNodes.get(random.nextInt(allNodes.size()));

        if (targetNode.type == NodeType.TERMINAL) {
            // For terminals, just replace with another terminal
            targetNode.value = createRandomTerminal().value;
        } else {
            // For functions, replace with another function of the same arity
            int arity = functions.get(targetNode.value);
            List<String> compatibleFunctions = functions.entrySet().stream()
                    .filter(e -> e.getValue() == arity)
                    .map(Map.Entry::getKey)
                    .toList();

            if (!compatibleFunctions.isEmpty()) {
                targetNode.value = compatibleFunctions.get(random.nextInt(compatibleFunctions.size()));
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
