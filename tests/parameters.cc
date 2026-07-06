#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <otter/parameters.h>

using namespace dealii;
using namespace Otter;

int
main()
{
  ScreenedPoissonParameters params;

  ParameterHandler prm;
  params.add_parameters(prm);

  prm.print_parameters(std::cout, ParameterHandler::OutputStyle::JSON);
}
