#####################
Qiskit noise learning
#####################

A Python toolkit for randomization-based quantum noise characterization.

.. note::

   **Alpha software.** This library is in the ``0.x`` stage of development and under active
   development. No part of the public interface is yet stable: while the major version is ``0``,
   expect breaking changes between releases and pin your dependency accordingly (for example,
   ``qiskit-noise-learning==0.1.*``). We do not currently issue deprecation warnings, but all
   changes are recorded in the changelog. See the `deprecation policy
   <https://github.com/Qiskit/qiskit-noise-learning/blob/main/DEPRECATION.md>`_ for details. All
   feedback is appreciated.

Interfaces and Documentation
----------------------------

This library has two levels of interface. The first is the low-level interface where the user
directly interacts with objects representing core concepts in noise learning, enabling custom design
of every aspect of a noise learning protocol. The second is a higher-level interface, the
``qiskit_noise_learning.protocols`` subpackage, whose ``prepare_learning_program`` and
``process_learning_result`` functions wrap a stock workflow into a single call on each side of
execution.

See the following guides for examples on how to use this package:

* :doc:`guides/learning_protocol` — end-to-end use of the protocol functions
* :doc:`guides/workflow` — step-by-step walkthrough of the internal pipeline


Contributing
------------

See the `contribution guidelines <https://github.com/Qiskit/qiskit-noise-learning/blob/main/CONTRIBUTING.md>`_ for details on developer setup, testing,
building the documentation, and the changelog workflow.


Citing this package
-------------------

If you use this package in your research, use the `CITATION.bib <https://github.com/Qiskit/qiskit-noise-learning/blob/main/CITATION.bib>`_ file in this project's repository to cite the appropriate reference(s).

.. toctree::
   :hidden:

   Documentation home <self>
   Installation instructions <install>
   Guides <guides/index>
   GitHub <https://github.com/Qiskit/qiskit-noise-learning>


.. toctree::
   :hidden:
   :caption: API reference

   Python API reference <https://quantum.cloud.ibm.com/docs/api/qiskit-noise-learning>

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
