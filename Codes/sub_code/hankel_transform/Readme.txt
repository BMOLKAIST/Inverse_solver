
				     MATLAB toolbox
		________________________________________________________

				    Hankel transform
		________________________________________________________


** Contents

	1. Introduction
	2. Requirements
	3. Implementation details
	4. References
	5. Copyright
	6. Warranty
	7. History
	8. Download
	9. Trademarks


** Publisher

	Marcel Leutenegger		marcel.leutenegger@epfl.ch
	EPFL STI IOA LOB
	BM 4.143			Tel:	+41 21 693 77 19
	Station 17
	CH-1015 Lausanne



1. Introduction

	The Hankel transform of order n transforms rotationally symetric inputs in a
	computationally efficient manner. In particular, the Hankel transform of order
	0 is equivalent to the two-dimensional Fourier transform of a rotationally
	symetric input. This package contains four implementations of the Hankel
	transform and the inverse Hankel transform, respectively.


2. Requirements

	• MATLAB 6.0 or newer.


3. Implementation details

	The package ships four implementations of the Hankel transform. "hat" and
	"ihat" perform the Hankel transform of order n with a direct integration
	using a matrix product. "ht" and "iht" perform the Hankel transform of order
	0 by integrating the Bessel kernel a priori. "dht" and "idht" implement the
	quasi-discrete Hankel transform of integer order n. And, last but not least,
	"fht" and "ifht" implement the quasi fast Hankel transform of order n.

	For more implementation details, please refer to the online documentation at

	    http://ioalinux1.epfl.ch/~mleutene/MATLABToolbox/HankelTransform.html


4. References

	• M. Guizar-Sicairos, J.C. Gutierrez-Vega, "Computation of quasi-discrete Hankel
	  transforms of integer order for propagating optical wave fields," J. Opt. Soc.
	  Am. A 21, 53-58 (2004).
	• D.G. Gardner, J.C. Gardner, G. Lausch, W.W. Meinke, "Method for the analysis
	  of multi-component exponential decays," J. Chem. Phys. 31, 987 (1959).
	• A.E. Siegmann, "Quasi fast Hankel transform," Opt. Lett. 1, 13-15 (1977).


5. Copyright

	These routines are published as freeware. The author reserves the right to modify
	any of the contained files.

	You are allowed to distribute these routines as long as you deliver for free the
	entire package.

		Path		File		Description

		/		dht.m		Quasi-discrete Hankel transform of integer
						order n
				dht.mat		Roots of Bessel functions of the first kind
				fht.m		Quasi fast Hankel transform of order n
				hat.m		Hankel transform of order n
				ht.m		Hankel transform of order 0
				idht.m		Inverse quasi-discrete Hankel transform of
						integer order n
				ifht.m		Inverse quasi fast Hankel transform of order n
				ihat.m		Inverse Hankel transform of order n
				iht.m		Inverse Hankel transform of order 0
				Readme.txt	This summary
		private/	frdr.m		Integrand
				JnRoots.m	Roots of Bessel functions of the first kind


6. Warranty

	Any warranty is strictly refused and you cannot anticipate any financial or
	technical support in case of malfunction or damage.

	Feedback and comments are welcome. I will try to track reported problems and
	fix bugs.


7. History

   • December 13, 2006
	Initial release

   • April 10, 2007
	Bug fix in "fht.m" thanks to Mark W. Sprague.


8. Download

	Optimized MATLAB routines are available online at:

		   http://ioalinux1.epfl.ch/~mleutene/MATLABToolbox/


	A summary is also published at MATLAB central:

			http://www.mathworks.com/matlabcentral/


9. Trademarks

	MATLAB is a registered trademark of The MathWorks, Inc.

		________________________________________________________

			    Site map • EPFL © 2006, Lausanne
				Webmaster • Dec 13, 2006
