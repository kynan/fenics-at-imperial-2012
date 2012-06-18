# More info at https://github.com/guard/guard#readme

require 'guard'
require 'guard/guard'
require 'guard/watcher'
require 'keydown'

module ::Guard
  class KeyDown < Guard
    def run_all
      run_on_changes(Watcher.match_files())
    end

    def run_on_changes(paths)
      #paths.each { |f| Keydown::Tasks.slides(f) }
      paths.each { |f| `keydown slides #{f}` }
    end
  end
end

guard 'keydown' do
  watch('slides.md')
end

guard 'sass', :input => 'css'
