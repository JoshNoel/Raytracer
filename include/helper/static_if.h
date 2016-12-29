#pragma once
#include <type_traits>
//source: http://stackoverflow.com/questions/37617677/implementing-a-compile-time-static-if-logic-for-different-string-types-in-a-co

namespace helper {
	//takes two functions t and f, calls t if true, f if false
	template<typename T, typename F>
	auto constexpr static_if(std::true_type, T t, F f) { return t; }

	template<typename T, typename F>
	auto constexpr static_if(std::false_type, T t, F f) { return f; }

	//calls t or f overload based on whether first template parameter is true or false
	template<bool B, typename T, typename F>
	auto constexpr static_if(T t, F f) { return static_if(std::integral_constant<bool, B>{}, t, f); }

	constexpr int dummy() { return 0; }

	template<bool B, typename T>
	auto constexpr static_if(T t) { return static_if(std::integral_constant<bool, B>{}, t, dummy()); }
}