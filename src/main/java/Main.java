public class Main {
    public static void main(String[] args) {
        try {
            HepatitisGP.main(args);
        } catch (Exception e) {
            System.err.println("Error running Hepatitis GP: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
