

import java.nio.DoubleBuffer;
import java.util.Arrays;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.DoubleByReference;

import lbfgs.Function;
import lbfgs.LbfgsLibrary;

/**
 * java call lbfgs
 *
 */
public class TestLbfgs {

	public double[] minimize(final Function function) {
		final int dim = function.getDimension();
		double[] x = new double[dim];

		double[] initialX = new double[dim];
		for (int n = 0; n < dim; n++) {
			initialX[n] = Math.random();
		}

		System.out.println("initialX=" + Arrays.toString(initialX));

		// DoubleBuffer paramX = LbfgsLibrary.INSTANCE.lbfgs_malloc(dim);
		// Pointer paramXPointer = paramX.getPointer();
		// for (long n = 0; n < dim; n++) {
		// paramX.getPointer().setDouble(n, initialX[(int)n]);
		// }
		DoubleBuffer paramX = DoubleBuffer.allocate(dim);
		for (int i = 0; i < initialX.length; i++) {
			paramX.put(i, initialX[i]);
		}
		LbfgsLibrary.lbfgs_evaluate_t f = new LbfgsLibrary.lbfgs_evaluate_t() {

			public double apply(Pointer instance, DoubleByReference x, DoubleByReference g,
				int n, double step) {
				double[] xArray = x.getPointer().getDoubleArray(0, dim);
				double[] gradientAt = function.gradientAt(xArray);
				Pointer gPointer = g.getPointer();
				gPointer.write(0, gradientAt, 0, gradientAt.length);
				return function.valueAt(xArray);
			}
		};

		int statusCode = LbfgsLibrary.INSTANCE.lbfgs(dim, paramX, null, f, null, null, null);
		// double[] finalXArray = paramX.getPointer().getDoubleArray(0, dim);
		//
		// double finalX = paramX.getValue();
		// System.out.println("x at min value=" + finalX);
		System.out.println("statusCode=" + statusCode);

		return x;
	}

	public static void main(String[] args) {
		TestLbfgs testLbfgs = new TestLbfgs();

		Function function1 = new Function() {

			public int getDimension() {
				return 1;
			}

			public double valueAt(double[] x) {
				System.out.println(Arrays.toString(x));
				return Math.pow(x[0] - 5, 2) + 1;
			}

			public double[] gradientAt(double[] x) {
				return new double[] {2 * (x[0] - 5)};
			}
		};


		Function function2 = new Function() {

			public int getDimension() {
				return 2;
			}

			public double valueAt(double[] x) {
				double v = Math.pow(x[0] - 5, 2) + Math.pow(x[1] - 3, 2) + 1;
				System.out.println("f\t" + Arrays.toString(x) + "=" + v);
				return v;
			}

			public double[] gradientAt(double[] x) {
				double[] g = new double[] {2 * (x[0] - 5), 2 * (x[1] - 3)};

				System.out.println("g\t" + Arrays.toString(x) + "=" + Arrays.toString(g));

				return g;
			}
		};

		testLbfgs.minimize(function2);

	}
}
