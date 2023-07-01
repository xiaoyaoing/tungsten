#include "Debug.hpp"

#include <iostream>

namespace Tungsten {

const   bool   DebugUtils::OnlyDirectLighting= true;
const bool DebugUtils::OnlyIndirectLighting = false;
const bool DebugUtils::OnlyOneThread = false;
const bool DebugUtils::OnlyShowNormal = false;

namespace DebugUtils {


void debugLog(const std::string &message)
{
    std::cout << message << std::endl;
}

}

}
