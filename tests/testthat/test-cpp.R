context("C++")

set.seed(1)
test_that("Catch unit tests pass", {
    expect_cpp_tests_pass("mixprobit")
})
