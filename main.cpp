#include "pr.hpp"
#include <iostream>
#include <memory>
#include <mutex>

class BlueNoiseGenerator
{
public:
	BlueNoiseGenerator()
	{
	}
	void allocate( int size )
	{
		_size = size;
		_values.resize( _size * _size );

		pr::Xoshiro128StarStar random;
		for ( int i = 0; i < _values.size(); ++i )
		{
			_values[i] = random.uniformi() % 256;
		}
	}
	double E() const
	{
		double e = 0;

		float sigma_i = 2.1f;
		std::mutex m;

		pr::ParallelFor( _values.size(), [&]( int i ) {
			double localE = 0;
			for ( int j = 0; j < _values.size(); ++j )
			{
				if ( i <= j )
				{
					break;
				}

				int x0 = i % _size;
				int y0 = i / _size;
				int x1 = j % _size;
				int y1 = j / _size;
				int dx = x1 - x0;
				int dy = y1 - y0;

				if ( _size / 2 < std::abs( dx ) )
				{
					if ( dx < 0 )
					{
						std::swap( x1, x0 );
					}
					x0 += _size;
					dx = x1 - x0;
				}
				if ( _size / 2 < std::abs( dy ) )
				{
					if ( dy < 0 )
					{
						std::swap( y1, y0 );
					}
					y0 += _size;
					dy = y1 - y0;
				}

				localE += std::exp( -( dx * dx + dy * dy ) / sigma_i - sqrt( fabs( (int)_values[i] - (int)_values[j] ) ) );
			}
			std::lock_guard<std::mutex> lc( m );
			e += localE;
		} );

		return e;
	}
	void apply( pr::Image2DRGBA32& image ) const
	{
		image.allocate( _size, _size );
		for ( int j = 0; j < _size; ++j )
		{
			for ( int i = 0; i < _size; ++i )
			{
				float value = _values[_size * j + i];
				image( i, j ) = glm::vec4( value, value, value, 1.0f );
			}
		}
	}
	void apply( pr::Image2DMono8& image ) const
	{
		image.allocate( _size, _size );
		for ( int j = 0; j < _size; ++j )
		{
			for ( int i = 0; i < _size; ++i )
			{
				image( i, j ) = _values[_size * j + i];
			}
		}
	}

	void step()
	{
		double currentE = E();
		for ( int i = 0; i < 16; ++i )
		{
			int a = _random.uniformi() % _values.size();
			int b = _random.uniformi() % _values.size();
			if ( a == b )
			{
				continue;
			}

			std::swap( _values[a], _values[b] );
			double newE = E();
			if ( newE < currentE )
			{
				printf( "flipped %f -> %f (%d, %d)\n", currentE, newE, a, b );
				currentE = newE;
			}
			else
			{
				std::swap( _values[a], _values[b] );
			}
		}
	}

private:
	int _size = 0;
	std::vector<uint8_t> _values;
	pr::Xoshiro128StarStar _random;
};

int main()
{
	using namespace pr;

	SetDataDir( ExecutableDir() );

	Config config;
	config.ScreenWidth = 1920;
	config.ScreenHeight = 1080;
	config.SwapInterval = 1;
	Initialize( config );

	Camera3D camera;
	camera.origin = {4, 4, 4};
	camera.lookat = {0, 0, 0};
	camera.zUp = true;

	int Size = 64;
	BlueNoiseGenerator bluenoise;
	bluenoise.allocate( Size );

	pr::Image2DMono8 imageOut;
	pr::ITexture* texture = CreateTexture();

	double e = GetElapsedTime();

	while ( pr::NextFrame() == false )
	{
		bluenoise.step();
		bluenoise.apply( imageOut );
		texture->upload( imageOut );

		if ( IsImGuiUsingMouse() == false )
		{
			UpdateCameraBlenderLike( &camera );
		}

		ClearBackground( 0.1f, 0.1f, 0.1f, 1 );

		BeginCamera( camera );

		PushGraphicState();

		DrawGrid( GridAxis::XY, 1.0f, 10, {128, 128, 128} );
		DrawXYZAxis( 1.0f );

		PopGraphicState();
		EndCamera();

		BeginImGui();

		ImGui::SetNextWindowSize( {500, 800}, ImGuiCond_Once );
		ImGui::Begin( "Panel" );
		ImGui::Text( "fps = %f", GetFrameRate() );

		ImGui::Image( texture, ImVec2( (float)texture->width() * 3, (float)texture->height() * 3 ) );

		if ( ImGui::Button( "Save" ) )
		{
			imageOut.save( "../bluenoise.png" );
		}

		ImGui::End();

		EndImGui();
	}

	pr::CleanUp();
}
