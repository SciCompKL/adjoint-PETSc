#include <gtest/gtest.h>

#include <petscsys.h>

int main(int argc, char **argv) {

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  testing::InitGoogleTest(&argc, argv);

  auto result = RUN_ALL_TESTS();

  PetscCall(PetscFinalize());
  return result;
}
