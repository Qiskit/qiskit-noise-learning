Added :func:`~.build_with_prep_noise` and :func:`~.inject_prep_noise`, which inject
state-preparation Pauli-Lindblad noise in a boxed circuit by modifying the
samplex. This avoids the redundant layer of 1Q gates previously included in the
definition of state-prep. A noise map can be supplied at sampling time as usual, with
its sampled Pauli signs appended to the ``pauli_signs`` output.
