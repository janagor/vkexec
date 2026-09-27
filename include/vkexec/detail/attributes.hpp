#ifndef VKEXEC_DETAIL_ATTRIBUTES_HPP
#define VKEXEC_DETAIL_ATTRIBUTES_HPP

#ifdef _MSC_VER
#define VKEXEC_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define VKEXEC_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

#endif
