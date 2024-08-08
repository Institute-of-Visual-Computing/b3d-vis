#pragma once

#include <initializer_list>
#include <map>
#include <optional>
#include <ranges>
#include <string>


#ifdef B3D_USE_NLOHMANN_JSON
	#include <nlohmann/json.hpp>
#endif

namespace b3d::tools::sofia
{
	// extern const std::map<std::string, std::string> DEFAULT_PARAMS;
	inline const std::map<std::string, std::string> DEFAULT_PARAMS = {
		// Global settings
		{ "pipeline.verbose", "false" },
		{ "pipeline.pedantic", "true" },
		{ "pipeline.threads", "0" },

		// Input
		{ "input.data", "" },
		{ "input.region", "" },
		{ "input.gain", "" },
		{ "input.noise", "" },
		{ "input.weights", "" },
		{ "input.primaryBeam", "" },
		{ "input.mask", "" },
		{ "input.invert", "false" },

		// Flagging
		{ "flag.region", "" },
		{ "flag.catalog", "" },
		{ "flag.radius", "5" },
		{ "flag.auto", "false" },
		{ "flag.threshold", "5.0" },
		{ "flag.log", "false" },

		// Continuum subtraction
		{ "contsub.enable", "false" },
		{ "contsub.order", "0" },
		{ "contsub.threshold", "2.0" },
		{ "contsub.shift", "4" },
		{ "contsub.padding", "3" },

		// Noise scaling
		{ "scaleNoise.enable", "false" },
		{ "scaleNoise.mode", "spectral" },
		{ "scaleNoise.statistic", "mad" },
		{ "scaleNoise.fluxRange", "negative" },
		{ "scaleNoise.windowXY", "25" },
		{ "scaleNoise.windowZ", "15" },
		{ "scaleNoise.gridXY", "0" },
		{ "scaleNoise.gridZ", "0" },
		{ "scaleNoise.interpolate", "false" },
		{ "scaleNoise.scfind", "false" },

		// Ripple filter
		{ "rippleFilter.enable", "false" },
		{ "rippleFilter.statistic", "median" },
		{ "rippleFilter.windowXY", "31" },
		{ "rippleFilter.windowZ", "15" },
		{ "rippleFilter.gridXY", "0" },
		{ "rippleFilter.gridZ", "0" },
		{ "rippleFilter.interpolate", "false" },

		// S+C finder
		{ "scfind.enable", "true" },
		{ "scfind.kernelsXY", "0, 3, 6" },
		{ "scfind.kernelsZ", "0, 3, 7, 15" },
		{ "scfind.threshold", "5.0" },
		{ "scfind.replacement", "2.0" },
		{ "scfind.statistic", "mad" },
		{ "scfind.fluxRange", "negative" },

		// Threshold finder
		{ "threshold.enable", "false" },
		{ "threshold.threshold", "5.0" },
		{ "threshold.mode", "relative" },
		{ "threshold.statistic", "mad" },
		{ "threshold.fluxRange", "negative" },

		// Linker
		{ "linker.enable", "true" },
		{ "linker.radiusXY", "1" },
		{ "linker.radiusZ", "1" },
		{ "linker.minSizeXY", "5" },
		{ "linker.minSizeZ", "5" },
		{ "linker.maxSizeXY", "0" },
		{ "linker.maxSizeZ", "0" },
		{ "linker.minPixels", "0" },
		{ "linker.maxPixels", "0" },
		{ "linker.minFill", "0.0" },
		{ "linker.maxFill", "0.0" },
		{ "linker.positivity", "false" },
		{ "linker.keepNegative", "false" },

		// Reliability
		{ "reliability.enable", "false" },
		{ "reliability.parameters", "peak, sum, mean" },
		{ "reliability.threshold", "0.9" },
		{ "reliability.scaleKernel", "0.4" },
		{ "reliability.minSNR", "3.0" },
		{ "reliability.minPixels", "0" },
		{ "reliability.autoKernel", "false" },
		{ "reliability.iterations", "30" },
		{ "reliability.tolerance", "0.05" },
		{ "reliability.catalog", "" },
		{ "reliability.plot", "true" },
		{ "reliability.debug", "false" },

		// Mask dilation
		{ "dilation.enable", "false" },
		{ "dilation.iterationsXY", "10" },
		{ "dilation.iterationsZ", "5" },
		{ "dilation.threshold", "0.001" },

		// Parameterisation
		{ "parameter.enable", "true" },
		{ "parameter.wcs", "true" },
		{ "parameter.physical", "false" },
		{ "parameter.prefix", "SoFiA" },
		{ "parameter.offset", "false" },

		// Output
		{ "output.directory", "" },
		{ "output.filename", "" },
		{ "output.writeCatASCII", "true" },
		{ "output.writeCatXML", "true" },
		{ "output.writeCatSQL", "false" },
		{ "output.writeNoise", "false" },
		{ "output.writeFiltered", "false" },
		{ "output.writeMask", "false" },
		{ "output.writeMask2d", "false" },
		{ "output.writeRawMask", "false" },
		{ "output.writeMoments", "false" },
		{ "output.writeCubelets", "false" },
		{ "output.writePV", "false" },
		{ "output.writeKarma", "false" },
		{ "output.marginCubelets", "10" },
		{ "output.thresholdMom12", "0.0" },
		{ "output.overwrite", "true" }
	};

	/// \brief Parameters for executing SoFiA. Those Params are copied from the SoFiA source code/wiki.
	class SofiaParams
	{
		public:
		SofiaParams() = default;
		SofiaParams(std::initializer_list<std::pair<const std::string, std::string>> params)
		{
			for (const auto& param : params)
			{
				if (isKeyValid(param.first))
				{
					params_.insert(param);
				}
			}
		}

		SofiaParams(const std::map<std::string, std::string> keyValueMap)
		{
			for (const auto& key : keyValueMap | std::views::keys)
			{
				if (!isKeyValid(key))
				{
					params_.erase(key);
				}
			}
		}


		//// \brief Check if the key is valid based on the default parameters
		static auto isKeyValid(const std::string& key) -> bool
		{
			return DEFAULT_PARAMS.contains(key);
		}

		//// \brief Concatenate key and value for CLI
		static inline auto concatCLI(const std::string& key, const std::string& value) -> std::string
		{
			return key + "=" + value;
		}

		/// \brief Set or override a parameter
		/// \param key Key
		/// \param value Value as string
		/// \return true if the parameter was set, false otherwise. key must be one of the valid parameters.
		auto setOrReplace(const std::string& key, const std::string& value) -> bool
		{
			if (!isKeyValid(key))
			{
				return false;
			}
			params_.erase(key);
			return params_.insert({ key, value }).second;
		}

		/// \brief Get Value as string
		/// \param key Key
		/// \return Value as string or std::nullopt if the key is not found
		auto getStringValue(const std::string& key) -> std::optional<std::string>
		{
			return params_.contains(key) ? std::make_optional(params_.at(key)) : std::nullopt;
		}

		/// \brief Check if the given key is present
		/// \param key Key
		/// \return True if the key is present, false otherwise
		auto containsKey(const std::string& key) const -> bool
		{
			return params_.contains(key);
		}

		/// \brief Builds the command line argument. Empty string if key is not found
		/// \param key Key
		/// \return CLI ready string or empty string
		auto getKeyValueCliArgument(const std::string& key) const -> std::string
		{
			const auto& it = params_.find(key);
			if (it == params_.end())
			{
				return "";
			}

			return concatCLI(it->first, it->second);
		}

		/// \brief Builds the command line arguments for all keys
		///	\param useDefaultIfKeyMissing If true, it will use the default value if the key is not found
		/// \return CLI ready string or empty string
		auto buildCliArguments(bool useDefaultIfKeyMissing = true) const -> std::vector<std::string>
		{
			std::vector<std::string> cliArgs = {};
			// TODO: ..par-file?

			// When .par-file is used do not use the default values. Just use the values from the map.

			for (const auto& [key, defaultValue] : DEFAULT_PARAMS)
			{
				const auto& it = params_.find(key);
				if (it != params_.end() && !it->second.empty())
				{
					cliArgs.emplace_back(concatCLI(it->first, it->second));
				}
				else if (useDefaultIfKeyMissing && !defaultValue.empty())
				{
					cliArgs.emplace_back(concatCLI(key, defaultValue));
				}
			}
			return cliArgs;
		}

		/// \brief Builds the command line arguments for all keys
		///	\param useDefaultIfKeyMissing If true, it will use the default value if the key is not found
		/// \return CLI ready string or empty string
		auto buildCliArgumentsAsString(bool useDefaultIfKeyMissing = true) const -> std::string
		{
			std::string cliArgs = "";
			// TODO: ..par-file?

			// When .par-file is used do not use the default values. Just use the values from the map.

			for (const auto& [key, defaultValue] : DEFAULT_PARAMS)
			{
				const auto& it = params_.find(key);
				if (it != params_.end() && !it->second.empty())
				{
					cliArgs += concatCLI(it->first, it->second) + " ";
				}
				else if (useDefaultIfKeyMissing && !defaultValue.empty())
				{
					cliArgs += concatCLI(key, defaultValue) + " ";
				}
			}
			return cliArgs;
		}
		
		auto begin() const
		{
			return params_.begin();
		}

		auto end() const
		{
			return params_.end();
		}

		private:
			std::map<std::string, std::string> params_{};

		#ifdef B3D_USE_NLOHMANN_JSON
				NLOHMANN_DEFINE_TYPE_INTRUSIVE(SofiaParams, params_);
		#endif
	};

} // namespace b3d::tools::sofia
